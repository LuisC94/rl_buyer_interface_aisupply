import numpy as np
import torch

class ReplayBuffer:
    """
    Buffer de Replay Circular de alta performance para algoritmos Off-Policy (DSAC).
    Armazena transições (s, a, r, s', done) em arrays NumPy pré-alocados para máxima eficiência.
    """
    def __init__(self, state_dim, action_dim, max_size=500_000, device="cpu"):
        self.max_size = int(max_size)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """ Insere uma transição única no buffer circular. """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """ Amostra aleatória uniforme de um mini-batch convertido para tensores PyTorch. """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        )

    def __len__(self):
        return self.size

