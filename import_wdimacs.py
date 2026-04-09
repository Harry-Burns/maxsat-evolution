import numpy as np

from satisfiability import satisfiability

def import_wdimacs(file):
    clauses = []

    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "c":
                continue

            if line[0] == "p":
                parts = line.split()
                N = int(parts[2])
                M = int(parts[3])
                x = True if len(parts) == 5 and parts[4] == "x" else False
                continue
            
            parts = line.split()
            clause = []
            
            for p in parts[1:]:
                if p == "0":
                    break

                clause.append(int(p))
            clauses.append(np.array(clause, dtype=int))

    assert len(clauses) == M

    return {"clauses": clauses, "N": N, "M": M, "x": x}