def is_done(prev_state, state):
    if state.get("race_finished"):
        return True

    # рестарт / телепорт
    if state["speed_norm"] < 0.1 and prev_state["speed_norm"] > 10:
        return True

    # падение с трассы / баг
    if abs(state["position"][1]) > 50:
        return True

    return False
