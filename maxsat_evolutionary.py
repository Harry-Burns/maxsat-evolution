import time
import numpy as np
from numba import njit


from import_wdimacs import import_wdimacs



@njit
def _fitness(pop, clause_lits, clause_offsets):
    n_pop = pop.shape[0]
    n_clauses = len(clause_offsets) - 1
    scores = np.zeros(n_pop, dtype=np.int32)
    for i in range(n_pop):
        for c in range(n_clauses):
            for k in range(clause_offsets[c], clause_offsets[c + 1]):
                v = clause_lits[k]
                if v > 0:
                    if pop[i, v - 1]:
                        scores[i] += 1
                        break
                else:
                    if not pop[i, -v - 1]:
                        scores[i] += 1
                        break
    return scores


class GeneticAlgorithm:
    def __init__(self, wdimacs):
        #np.random.seed(21)
        self.N = wdimacs['N']
        self.M = wdimacs['M']
        self.clauses = wdimacs['clauses']

        self.N_pop = 1024
        self.N_parents = 256
        self.N_child = 1024
        self.N_elite = 32

        self.pm = 1 / self.N

        # ----- Heuristics
        pos_w, neg_w = self.generate_heuristic_weights()
        self.pos_w = pos_w; self.neg_w = neg_w
        # ----------------


        # ----- Quick fitness
        lits = []
        offsets = [0]
        for clause in self.clauses:
            lits.extend(clause)
            offsets.append(len(lits))
        self.clause_lits = np.array(lits, dtype=np.int32)
        self.clause_offsets = np.array(offsets, dtype=np.int32)
        # -------------------
        


    def run(self, time_budget):
        start_time = time.time()

        pop = self.initialise()
        fit = self.fitness(pop)

        gen = 0
        while time.time() - start_time < time_budget:
            parents = self.tournament_selection(pop, fit, k=4)

            children = self.crossover(parents)
            pop_child = self.mutation(children, self.pm)

            #pop_child = self.heuristic_improvement_operator(pop_child)

            fit_child = self.fitness(pop_child)
            pop, fit = self.replacement(pop, pop_child, fit, fit_child)

            gen += 1
    

        best_i = np.argmax(fit)
        
        runtime = gen * max(self.N_child, self.N_pop)
        result = {'t': runtime, 'nsat': fit[best_i], 'xbest': pop[best_i], 'gen': gen, 'pop_size': max(self.N_child, self.N_pop)}
        return result


    def initialise(self):
        total_w = self.pos_w - self.neg_w
        init_pop = np.random.rand(self.N_pop, self.N) < total_w
        return init_pop
    

    def initialise(self, temperature=1.0):
        def _default_prob_fn(weights, temperature=1.0):
            return 1.0 / (1.0 + np.exp(-weights / temperature))

        total_w = self.pos_w - self.neg_w  # shape: (N,)
        probs = _default_prob_fn(total_w, temperature)

        init_pop = np.random.rand(self.N_pop, self.N) < probs

        return init_pop


    #def heuristic_improvement_operator(self, p):
    #    return p

    def fitness(self, p):
        return _fitness(p.astype(bool), self.clause_lits, self.clause_offsets)

    def tournament_selection(self, p, fit, k=2):
        N = p.shape[0]
        parents = []

        for _ in range(self.N_parents):
            idx = np.random.randint(0, N, size=k)
            winner = idx[np.argmax(fit[idx])]
            parents.append(p[winner])

        return np.array(parents)
        
    def ranking_selection(self, p, fit, s=1.5):
        N = p.shape[0]

        order = np.argsort(fit)
        ranks = np.empty(N, dtype=int)
        ranks[order] = np.arange(1, N + 1)

        probs = ((2 - s) / N) + (2 * (ranks - 1) * (s - 1)) / (N * (N - 1))
        probs = probs / probs.sum()

        chosen = np.random.choice(N, size=self.N_parents, replace=True, p=probs)
        return p[chosen]        


    def mutation(self, p, pm):
        bit_flips = np.random.rand(*p.shape) < pm
        p[bit_flips] = ~p[bit_flips]
        return p


    def crossover(self, parents):
        children = []

        def parent_crossover(x1, x2):
            mask = np.random.rand(self.N) < 0.5
            c1 = x1.copy()
            c2 = x2.copy()
            c1[mask] = x2[mask]
            c2[mask] = x1[mask]
            return c1, c2

        while len(children) < self.N_child:
            i, j = np.random.choice(len(parents), 2, replace=False)
            c1, c2 = parent_crossover(parents[i], parents[j])

            children.append(c1)

            if len(children) < self.N_child:
                children.append(c2)

        return np.array(children)


    def replacement(self, p, p_child, fit, fit_child):
        p_rank = np.argsort(fit)[::-1]

        elite_idx = p_rank[:self.N_elite]
        remaining = self.N_pop - self.N_elite
        non_elite_idx = p_rank[self.N_elite:]

        p_comb = np.vstack([p[non_elite_idx], p_child])
        fit_comb = np.concatenate([fit[non_elite_idx], fit_child])

        p_comb_rank = np.argsort(fit_comb)[::-1]
        selected_idx = p_comb_rank[:remaining]

        p = np.vstack([p[elite_idx], p_comb[selected_idx]])
        fit = np.concatenate([fit[elite_idx], fit_comb[selected_idx]])

        return p, fit


    def generate_heuristic_weights(self):
        pos_weights = np.zeros(self.N, dtype=int)
        neg_weights = np.zeros(self.N, dtype=int)

        for clause in self.clauses:
            for v in clause:
                i = abs(v) - 1
                if v > 0:
                    pos_weights[i] += 1
                else:
                    neg_weights[i] += 1

        return pos_weights, neg_weights           

















