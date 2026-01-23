def ppo_update(model, buffer, optimizer, gamma=0.99, eps_clip=0.2):
    # compute returns & advantages
    returns = []
    G = 0
    for r, d in zip(reversed(buffer.rewards), reversed(buffer.dones)):
        if d: G = 0
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    values = torch.cat(buffer.values).squeeze()
    advantages = returns - values.detach()

    obs = torch.stack(buffer.obs)
    actions = torch.stack(buffer.actions)
    old_logprobs = torch.stack(buffer.logprobs)

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
