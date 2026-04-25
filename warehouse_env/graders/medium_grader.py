def grade(done, collisions):
    score = 0.5 if done else 0
    score -= collisions * 0.1
    return max(0, min(1, score))
