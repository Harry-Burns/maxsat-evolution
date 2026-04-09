import numpy as np

# Assumes inputs are correctly formatted
def satisfiability(assignment: np.ndarray, clause: np.ndarray):
    assert np.max(np.abs(clause)) <= len(assignment)

    for vi in clause:
        idx = abs(vi) - 1
        if assignment[idx] == (vi > 0):
            return 1
    return 0



