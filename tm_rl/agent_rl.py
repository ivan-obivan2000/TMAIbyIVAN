# псевдокод логики

obs = obs_vec(state)
mu, std, value = model(obs)
action = sample(mu, std)

apply_action(action)

reward = compute_reward(prev_state, state, action)
done = is_done(prev_state, state)

buffer.add(obs, action, logprob, reward, value, done)

if buffer_size >= N:
    ppo_update(...)
    buffer.clear()
