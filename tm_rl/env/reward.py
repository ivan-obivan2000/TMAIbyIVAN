def compute_reward(prev_r, r, action):
    reward = 0.0

    # скорость
    reward += r["speed_norm"] * 0.01

    # штраф за боковое скольжение
    reward -= abs(r.get("sync", {}).get("speed_sideward", 0.0)) * 0.05

    # штраф за резкий руль
    steer = action[0]
    reward -= abs(steer) * 0.02

    # штраф за скольжение
    if r.get("car", {}).get("is_sliding"):
        reward -= 0.5

    return reward
