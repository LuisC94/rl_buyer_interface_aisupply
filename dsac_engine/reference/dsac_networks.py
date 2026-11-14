import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

def weights_init_xavier(m):
    """ Inicialização de pesos Xavier/Glorot para estabilidade """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class SquashedGaussianActor(nn.Module):
    """
    Ator Estocástico Gaussiano com corte Tanh para o DSAC.
    Recebe as 30 variáveis de estado e produz ações contínuas em [-1, 1].
    """
    def __init__(self, state_dim, action_dim=1, hidden_dim=256):
        super(SquashedGaussianActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_xavier)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """ Amostragem com reparametrização e correção jacobiana da log-probabilidade """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        u = normal.rsample()
        action = torch.tanh(u)

        log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean)

        return action, log_prob, deterministic_action


class TwinQuantileCritic(nn.Module):
    """
    Crítico Distribucional Duplo (Twin Quantile Critics - Z1 e Z2) para o DSAC.
    Em vez de prever 1 escalar de valor médio, cada rede prevê N quantis (ex: 32)
    que descrevem toda a curva de densidade de probabilidade de retornos futuros.
    Entrada: Estado (30) + Ação (1) = 31 dimensões.
    Saída: Tensor [Batch, Num_Quantiles] para Z1 e Z2.
    """
    def __init__(self, state_dim, action_dim=1, hidden_dim=256, num_quantiles=32):
        super(TwinQuantileCritic, self).__init__()
        self.num_quantiles = num_quantiles

        # Rede Z1 (Distribuição Quântica 1)
        self.z1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quantiles)
        )

        # Rede Z2 (Distribuição Quântica 2)
        self.z2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quantiles)
        )

        self.apply(weights_init_xavier)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        z1 = self.z1_net(sa)
        z2 = self.z2_net(sa)
        return z1, z2

    def z1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.z1_net(sa)

