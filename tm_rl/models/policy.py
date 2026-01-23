import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim: int = 3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # actor (раздельные головы)
        self.mu_steer = nn.Linear(256, 1) # steer
        self.mu_tb = nn.Linear(256, 2)    # gas, brake
        
        # critic
        self.value = nn.Linear(256, 1)

        # log_std общий параметр
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Инициализация весов (опционально, но полезно)
        with torch.no_grad():
            self.mu_steer.bias.fill_(0.0)
            self.mu_tb.bias[0].fill_(1.5)   # throttle bias
            self.mu_tb.bias[1].fill_(-2.0)  # brake bias

    def forward(self, x):
        h = self.shared(x)

        # Объединяем выходы: steer [-1, 1], gas/brake [0, 1]
        mu = torch.cat([
            torch.tanh(self.mu_steer(h)),
            torch.sigmoid(self.mu_tb(h))
        ], dim=1)

        std = self.log_std.exp()
        value = self.value(h)

        return mu, std, value