from math import sqrt


def compute_reward(prev_r, r, action, return_info=False):
    reward = 0.0
    info = {}

    # скорость
    speed_reward = r.get("speed_norm", 0.0) * 0.01
    reward += speed_reward
    info["speed"] = speed_reward

    # штраф за боковое скольжение
    side_slip_penalty = -abs(r.get("sync", {}).get("speed_sideward", 0.0)) * 0.05
    reward += side_slip_penalty
    info["side_slip"] = side_slip_penalty

    # штраф за резкий руль
    steer = float(action[0]) if action is not None else 0.0
    steer_penalty = -abs(steer) * 0.02
    reward += steer_penalty
    info["steer"] = steer_penalty

    # штраф за скольжение
    sliding_penalty = 0.0
    if r.get("car", {}).get("is_sliding"):
        sliding_penalty = -0.5
        reward += sliding_penalty
    info["sliding"] = sliding_penalty

    # вознаграждение за пройденное расстояние
    prev_pos = prev_r.get("position") or r.get("position") or (0.0, 0.0, 0.0)
    pos = r.get("position") or prev_pos
    dx = pos[0] - prev_pos[0]
    dy = pos[1] - prev_pos[1]
    dz = pos[2] - prev_pos[2]
    distance_delta = sqrt(dx * dx + dy * dy + dz * dz)
    distance_reward = distance_delta * 0.05
    reward += distance_reward
    info["distance"] = distance_reward
    info["distance_delta"] = distance_delta

    # штраф за время (чем быстрее, тем лучше)
    prev_time = int(prev_r.get("race_time_ms", r.get("race_time_ms", 0)))
    race_time_ms = int(r.get("race_time_ms", prev_time))
    delta_time_ms = max(0, race_time_ms - prev_time)
    time_penalty = -delta_time_ms * 0.001
    reward += time_penalty
    info["time"] = time_penalty
    info["delta_time_ms"] = delta_time_ms

    # бонус за финиш и время заезда
    finish_bonus = 0.0
    time_bonus = 0.0
    if r.get("race_finished"):
        target_time_ms = 60000
        time_bonus = max(0.0, (target_time_ms - race_time_ms) / 1000.0)
        finish_bonus = 50.0 + time_bonus
        reward += finish_bonus
    info["finish_bonus"] = finish_bonus
    info["time_bonus"] = time_bonus

    if return_info:
        info["total"] = reward
        return reward, info

    return reward
