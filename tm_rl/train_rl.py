"""PPO update step for Trackmania RL."""

from __future__ import annotations

from typing import Dict

import torch


def ppo_update(
    model,
    buffer,
    optimizer,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
) -> Dict[str, float]:
    if not buffer.rewards:
        return {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}
    # compute returns & advantages
    returns = []
    G = 0
    for r, d in zip(reversed(buffer.rewards), reversed(buffer.dones)):
        if d: G = 0
        G = r + gamma * G
        returns.insert(0, G)

    values = torch.cat(buffer.values).squeeze()
    device = values.device
    returns = torch.tensor(returns, dtype=values.dtype, device=device)
    advantages = returns - values.detach()

    obs = torch.stack(buffer.obs).to(device)
    actions = torch.stack(buffer.actions).to(device)
    old_logprobs = torch.stack(buffer.logprobs).to(device)
    if old_logprobs.dim() > 1:
        old_logprobs = old_logprobs.sum(dim=1)

    mu, std, new_values = model(obs)
    dist = torch.distributions.Normal(mu, std)
    new_logprobs = dist.log_prob(actions).sum(dim=1)

    ratio = (new_logprobs - old_logprobs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = (new_values.squeeze() - returns).pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
    }
