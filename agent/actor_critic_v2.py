import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorMLP(nn.Module):
    """
    The Actor (Intuition).
    Reads the State (Environment) and outputs a continuous Action (Order Quantity).
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorMLP, self).__init__()
        
        self.max_action = max_action
        
        # Dense layers for fast mapping of state -> action
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, action_dim)
        
        # --- Incerteza Dinâmica Percentual ---
        # Como o Cérebro agora opera em percentuais Puros Universais, podemos
        # nascer gentilmente em 0.0 (Gera um desvio Padrão Puro de 1.0 = 100% do range).
        self.log_std = nn.Parameter(torch.full((1, action_dim), 0.0))
        
    def forward(self, state):
        # Activation ReLU is standard for hidden layers
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        # Sigmoid obriga o Robô a comunicar o "Botão do Volume" entre [0.0 e 1.0]
        action_mean_percent = torch.sigmoid(self.output_layer(x))
        
        return action_mean_percent, self.log_std


class CriticMLP(nn.Module):
    """
    The Critic (The Judge / Financial Forecaster).
    Reads both the State AND the Actor's Action.
    Outputs the Q-Value (The predicted Profit/Loss of that specific decision).
    """
    def __init__(self, state_dim):
        super(CriticMLP, self).__init__()
        
        # --- State Pipeline ---
        self.state_layer = nn.Linear(state_dim, 64)
        self.layer1 = nn.Linear(64, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        
        # Output is a single linear float (The Baseline Value / V-Value)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, state):
        # 1. Process State
        x = F.relu(self.state_layer(state))
        
        # 2. Deep Evaluation
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        # 3. Final V-Value (Baseline value of the state itself)
        v_value = self.output_layer(x)
        return v_value
