def grade(env):
    sm = env.state_manager

    score = 0.0

    # Completion — use completed_orders (accurate, not carrying-based)
    completion_ratio = len(sm.completed_orders) / max(len(sm.orders), 1)
    score += completion_ratio * 0.5

    # Efficiency
    score += max(0, 0.3 - sm.steps * 0.005)

    # Safety
    score += max(0, 0.2 - sm.collisions * 0.05)

    # Cooperation bonus — both agents contributed
    if sm.check_cooperation_bonus():
        score += 0.1

    return round(min(score, 1.0), 3)
