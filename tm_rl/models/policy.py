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

        # actor
        self.mu_steer = nn.Linear(256, 1)
        self.mu_tb = nn.Linear(256, 2)      # throttle, brake
        self.log_std = nn.Parameter(torch.zeros(3))

        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        

        # critic
        self.value = nn.Linear(256, 1)

        with torch.no_grad():
            self.mu_steer.bias.fill_(0.0)
            self.mu_tb.bias[0].fill_(1.5)   # throttle
            self.mu_tb.bias[1].fill_(-2.0) 

    def forward(self, x):
        h = self.shared(x)

        mu = torch.cat([
            torch.tanh(self.mu_steer(h)),   # [-1..1]
            torch.sigmoid(self.mu_tb(h))    # [0..1]
        ], dim=1)

        std = self.log_std.exp()
        value = self.value(h)

        return mu, std, value
