import argparse
import numpy as np


# region Question 3
def maxstat_evolution_str(wdimacs, time_budget, repetitions):
    from import_wdimacs import import_wdimacs
    from maxsat_evolutionary import GeneticAlgorithm

    wdimacs = import_wdimacs(wdimacs)

    results = []
    for _ in range(repetitions):
        ga = GeneticAlgorithm(wdimacs=wdimacs)
        r = ga.run(time_budget=time_budget)
        results.append(r)

    out = ""
    for r in results:
        out += f"{r['t']} {r['nsat']} ({wdimacs['M'] - r['nsat']}) {"".join([str(int(x)) for x in r['xbest']])}\n"

    return out
# endregion


# region Question 2
def wdimacs_str(assignment, file):
    from satisfiability import satisfiability
    from import_wdimacs import import_wdimacs
    wdimacs = import_wdimacs(file)

    clauses = wdimacs["clauses"]
    assign = np.fromiter((c == "1" for c in assignment), dtype=bool)

    assert len(assign) == wdimacs["N"]
    
    clause_satisfiability = [satisfiability(assign, c) for c in clauses]

    return sum(clause_satisfiability)
# endregion


# region Question 1
def satisfiability_str(assignment: str, clause: str):
    from satisfiability import satisfiability
    assignment_np = np.fromiter((c == "1" for c in assignment), dtype=bool)

    vals = []
    for v in clause.split()[1:]:
        vi = int(v)
        if vi == 0:
            break
        vals.append(vi)

    clause_np = np.array(vals, dtype=int)

    return satisfiability(assignment_np, clause_np)
# endregion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-question", required=False, type=int)
    parser.add_argument("-clause", required=False, type=str)
    parser.add_argument("-assignment", required=False, type=str)
    parser.add_argument("-wdimacs", required=False, type=str)
    parser.add_argument("-time_budget", required=False, type=int)
    parser.add_argument("-repetitions", required=False, type=int)

    args = parser.parse_args()

    if args.question == 1:
        out = satisfiability_str(args.assignment, args.clause)

    elif args.question == 2:
        out = wdimacs_str(args.assignment, args.wdimacs)

    elif args.question == 3:
        out = maxstat_evolution_str(args.wdimacs, args.time_budget, args.repetitions)

    else:
        out = "Invalid 'question' parameter. Exiting..."

    print(out)