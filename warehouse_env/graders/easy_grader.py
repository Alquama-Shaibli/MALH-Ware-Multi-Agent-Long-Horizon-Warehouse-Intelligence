def grade(done, steps):
    if done:
        return min(1.0, 1 - steps * 0.01)
    return 0.0
