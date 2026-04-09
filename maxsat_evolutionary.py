import time
import numpy as np



from import_wdimacs import import_wdimacs





class GeneticAlgorithm:
    def __init__(self, wdimacs):
        #np.random.seed(21)
        self.N = wdimacs['N']
        self.clauses = wdimacs['clauses']

        self.N_pop = 512
        self.N_parents = 128
        self.N_child = 512

        self.N_elite = 16

    def run(self, time_budget):
        start_time = time.time()

        pop = np.random.rand(self.N_pop,self.N) < 0.5
        fit = self.fitness(pop)

        gen = 0
        while time.time() - start_time < time_budget:
            parents = self.selection(pop, fit)
            children = self.crossover(parents)
            pop_child = self.mutation(children, 0.05)
            fit_child = self.fitness(pop_child)
            pop, fit = self.replacement(pop, pop_child, fit, fit_child)

            gen += 1
    

        best_i = np.argmax(fit)
        
        runtime = gen * max(self.N_child, self.N_pop)
        result = {'t': runtime, 'nsat': fit[best_i], 'xbest': pop[best_i]}
        return result



    def fitness(self, p):
        pop = p.astype(bool)
        scores = np.zeros(p.shape[0], dtype=np.int32)

        for clause in self.clauses:
            sat = np.zeros(p.shape[0], dtype=bool)
            for v in clause:
                i = abs(v) - 1
                sat |= pop[:, i] if v > 0 else ~pop[:, i]
            scores += sat
        return scores


    def selection(self, p, fit):
        N = p.shape[0]
        parents = []

        for _ in range(self.N_parents):
            i, j = np.random.randint(0, N, 2)
            winner = i if fit[i] > fit[j] else j
            parents.append(p[winner])
        return np.array(parents)


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




















