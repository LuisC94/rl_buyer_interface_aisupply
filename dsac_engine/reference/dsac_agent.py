import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dsac_networks import SquashedGaussianActor, TwinQuantileCritic

class DSACAgent:
    """
    Agente Distributional Soft Actor-Critic (DSAC / DSAC-T).
    Substitui a previsão escalar média por uma distribuição contínua de quantis de probabilidade.
    
    Características:
    - Ator Estocástico Gaussiano com Tanh
    - Dois Críticos Distribucionais Quânticos (Twin Quantile Critics Z1 e Z2)
    - Função de Perda Quantile Huber Loss (Pinball Loss)
    - Parâmetro de Aversão ao Risco (Risk Factor / CVaR) para controlar penalização de perdas extremas
    - Ajuste automático de Temperatura de Entropia (Alpha)
    """
    def __init__(
        self,
        state_dim=30,
        action_dim=1,
        gamma=0.99,
        tau=0.005,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        num_quantiles=32,
        risk_factor=1.0,
        kappa=1.0,
        target_entropy=None,
        hidden_dim=256,
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.risk_factor = float(risk_factor)  # 1.0 = Média Neutra | 0.8 = Foco nos 80% piores cenários (Anti-Desperdício)
        self.kappa = float(kappa)              # Limiar da Huber Loss

        # Posições fracionárias dos quantis tau_i = (2i - 1) / (2N) para regressão quântica
        tau_fractions = (torch.arange(num_quantiles, dtype=torch.float32, device=self.device) + 0.5) / num_quantiles
        self.tau_hat = tau_fractions.view(1, num_quantiles, 1)

        # 1. Redes Neuronais
        self.actor = SquashedGaussianActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQuantileCritic(state_dim, action_dim, hidden_dim, num_quantiles).to(self.device)
        self.critic_target = TwinQuantileCritic(state_dim, action_dim, hidden_dim, num_quantiles).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # 2. Otimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 3. Temperatura da Entropia Automática (Alpha)
        # Calibrada para retalho / supply chain (-2.5 * dim em vez de -1.0 * dim para precisão operacional)
        if target_entropy is None:
            self.target_entropy = -2.5 * float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        """ Seleciona a ação proposta pelo Ator. Retorna array em [-1, 1]. """
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)

            action, _, deterministic_action = self.actor.sample(state_tensor)
            action_to_use = deterministic_action if evaluate else action
            return action_to_use.cpu().numpy().flatten()

    def _quantile_huber_loss(self, z_pred, y_target):
        """
        Cálculo vetorizado da Quantile Huber Loss (Pinball Loss assimétrica):
        z_pred:   [Batch, N] (Previsão de quantis)
        y_target: [Batch, N] (Alvo quântico ordenado da equação de Bellman)
        """
        # Expandir dimensões para diferença de pares [Batch, N_pred, N_target]
        diff = y_target.unsqueeze(1) - z_pred.unsqueeze(2)
        abs_diff = diff.abs()

        # Huber loss suave
        huber = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa)
        )

        # Penalização assimétrica de acordo com o quantil tau_hat
        # Se o erro for negativo, peso = |tau - 1|; se positivo, peso = tau
        weight = torch.abs(self.tau_hat - (diff.detach() < 0).float())
        loss = (weight * huber).sum(dim=-1).mean()
        return loss

    def update(self, replay_buffer, batch_size=256):
        """ Executa 1 passo de otimização distribucional completa (Críticos, Ator, Alpha). """
        if len(replay_buffer) < batch_size:
            return None

        # 1. Amostrar transições do Buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 2. Distribuição Alvo de Bellman com Controlo de Incerteza
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_z1, target_z2 = self.critic_target(next_states, next_actions)

            # Combinação conservadora quântica entre as duas redes alvo
            target_z_min = torch.min(target_z1, target_z2) - (self.alpha * next_log_probs)

            # Equação de Bellman Distribucional: Y = r + gamma * (1 - done) * Z
            target_z = rewards + (1.0 - dones) * self.gamma * target_z_min
            # Ordenar os quantis do alvo para manter monotonicidade matemática
            target_z = torch.sort(target_z, dim=-1)[0]

        # 3. Otimização dos Críticos Quânticos (Z1 e Z2)
        current_z1, current_z2 = self.critic(states, actions)
        loss_z1 = self._quantile_huber_loss(current_z1, target_z)
        loss_z2 = self._quantile_huber_loss(current_z2, target_z)
        critic_loss = loss_z1 + loss_z2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 4. Otimização do Ator (com Sensibilidade a Risco opcional)
        new_actions, log_probs, _ = self.actor.sample(states)
        z1_pi, z2_pi = self.critic(states, new_actions)
        min_z_pi = torch.min(z1_pi, z2_pi)  # Shape [Batch, N]

        # Se risk_factor < 1.0 (ex: 0.8), otimiza os 80% piores quantis (CVaR Anti-Desperdício)
        if self.risk_factor < 1.0:
            k = max(1, int(self.num_quantiles * self.risk_factor))
            sorted_z = torch.sort(min_z_pi, dim=-1)[0]
            risk_aware_val = sorted_z[:, :k].mean(dim=-1, keepdim=True)
        else:
            risk_aware_val = min_z_pi.mean(dim=-1, keepdim=True)

        actor_loss = ((self.alpha.detach() * log_probs) - risk_aware_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 5. Ajuste Automático da Temperatura (Alpha)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 6. Atualização Suave das Redes Alvo (Polyak)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "loss_critic": float(critic_loss.item()),
            "loss_actor": float(actor_loss.item()),
            "loss_alpha": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(min_z_pi.mean().item())
        }

    def save(self, directory, filename_prefix="dsac"):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename_prefix}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename_prefix}_critic.pth"))
        torch.save({
            "log_alpha": self.log_alpha.detach().cpu(),
            "target_entropy": self.target_entropy,
            "risk_factor": self.risk_factor
        }, os.path.join(directory, f"{filename_prefix}_alpha.pth"))

    def load(self, directory, filename_prefix="dsac"):
        actor_path = os.path.join(directory, f"{filename_prefix}_actor.pth")
        critic_path = os.path.join(directory, f"{filename_prefix}_critic.pth")
        alpha_path = os.path.join(directory, f"{filename_prefix}_alpha.pth")

        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
        if os.path.exists(alpha_path):
            alpha_data = torch.load(alpha_path, map_location=self.device)
            self.log_alpha.data.copy_(alpha_data["log_alpha"].to(self.device))
            if "risk_factor" in alpha_data:
                self.risk_factor = alpha_data["risk_factor"]

