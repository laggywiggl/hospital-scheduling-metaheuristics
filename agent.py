import numpy as np
import random
import math
import pandas as pd
from collections import Counter

# =================================================================
# 1. DONNÉES ET GÉNÉRATEUR D'INSTANCES
# =================================================================
skills = ["C1", "C2", "C3", "C4", "C5", "C6"]
patterns = [
    [["C1","C1"], ["C1","C2"], ["C1","C3"], ["C1","C2","C2"], ["C4","C5","C5","C6"]],
    [["C2","C3"], ["C2","C3"], ["C2"], [], []],
    [["C3","C3"], ["C3"], [], [], []],
    [["C4","C4"], ["C5","C6"], ["C6","C6"], ["C4","C4"], ["C1","C2"]],
    [["C2","C2"], ["C5"], ["C5","C6"], ["C4","C5"], ["C3"]],
]

def generate_hospital_data(nb_patients):
    return {i: random.choice(patterns) for i in range(1, nb_patients + 1)}

# =================================================================
# 2. SIMULATEUR DE CALCUL DU CMAX
# =================================================================
def schedule_serial(sequence, patient_ops):
    time_skill = {s: 0 for s in skills}
    time_patient = {p: 0 for p in sequence}
    for p in sequence:
        ops = patient_ops[p]
        for op_skills in ops:
            if not op_skills: continue
            duration = len(op_skills)
            # Calcul du temps de démarrage basé sur la disponibilité des ressources et du patient
            start_time = max(time_patient[p], max([time_skill[s] for s in op_skills]))
            end_time = start_time + duration
            time_patient[p] = end_time
            for s in op_skills:
                time_skill[s] = end_time
    return max(time_patient.values())

# =================================================================
# 3. ESPACE DE MÉMOIRE PARTAGÉ (EMP)
# =================================================================
class SharedMemory:
    def __init__(self):
        self.best_global_order = None
        self.best_global_cmax = float('inf')

    def update(self, order, cmax):
        if cmax < self.best_global_cmax:
            self.best_global_cmax = cmax
            self.best_global_order = list(order)

# =================================================================
# 4. CLASSES D'AGENTS (METAHEURISTIQUES + RL + COLLAB)
# =================================================================
class MetaHeuristicAgent:
    def __init__(self, name, algo_type, initial_order, patient_ops):
        self.name = name
        self.algo_type = algo_type.upper()
        self.patient_ops = patient_ops
        self.current_order = list(initial_order)
        self.best_order = list(initial_order)
        self.best_cmax = schedule_serial(initial_order, patient_ops)
        
    def evaluate(self, sequence):
        return schedule_serial(sequence, self.patient_ops)

    def work(self, iterations=10):
        res_order, res_cmax = list(self.best_order), self.best_cmax
        if self.algo_type == 'TABOU':
            res_order, res_cmax = self._run_tabu(self.current_order, iterations)
        elif self.algo_type == 'AG':
            res_order, res_cmax = self._run_ga(iterations)
        elif self.algo_type == 'RS':
            res_order, res_cmax = self._run_rs(self.current_order, iterations)
        
        if res_cmax < self.best_cmax:
            self.best_cmax, self.best_order = res_cmax, list(res_order)
        return self.best_cmax

    def _run_tabu(self, order, iters):
        curr, best = list(order), list(order)
        best_c = self.evaluate(best)
        tabu_list = []
        for _ in range(iters):
            i, j = random.sample(range(len(curr)), 2)
            neighbor = list(curr)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            move = tuple(sorted((i, j)))
            if move not in tabu_list:
                c = self.evaluate(neighbor)
                if c < best_c: best_c, best = c, neighbor
                tabu_list.append(move)
                if len(tabu_list) > 5: tabu_list.pop(0)
                curr = neighbor
        return best, best_c

    def _run_ga(self, iters):
        pop = [random.sample(self.current_order, len(self.current_order)) for _ in range(10)]
        best_ind = list(self.current_order)
        best_c = self.evaluate(best_ind)
        for _ in range(iters):
            pop.sort(key=lambda x: self.evaluate(x))
            if self.evaluate(pop[0]) < best_c:
                best_c, best_ind = self.evaluate(pop[0]), list(pop[0])
            for i in range(1, 10):
                idx1, idx2 = random.sample(range(len(pop[i])), 2)
                pop[i][idx1], pop[i][idx2] = pop[i][idx2], pop[i][idx1]
        return best_ind, best_c

    def _run_rs(self, order, iters):
        T, alpha = 100.0, 0.95
        curr, best = list(order), list(order)
        c_curr, c_best = self.evaluate(curr), self.evaluate(curr)
        for _ in range(iters):
            i, j = random.sample(range(len(curr)), 2)
            neigh = list(curr)
            neigh[i], neigh[j] = neigh[j], neigh[i]
            c_n = self.evaluate(neigh)
            if c_n < c_curr or (T > 0 and random.random() < math.exp(min(700, (c_curr - c_n)/T))):
                curr, c_curr = neigh, c_n
                if c_curr < c_best: c_best, best = c_curr, list(curr)
            T *= alpha
        return best, c_best

class CollabRLAgent(MetaHeuristicAgent):
    def __init__(self, name, algo_type, initial_order, patient_ops, use_rl=True):
        super().__init__(name, algo_type, initial_order, patient_ops)
        self.use_rl = use_rl
        self.q_table = np.zeros(3) # Actions: 0=Swap, 1=Insertion, 2=Inversion
        self.epsilon, self.alpha_rl = 0.2, 0.1

    def apply_neighborhood(self, order, action):
        new_order = list(order)
        n = len(new_order)
        if action == 0: # Swap
            i, j = random.sample(range(n), 2)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif action == 1: # Insertion
            val = new_order.pop(random.randint(0, n-1))
            new_order.insert(random.randint(0, n-1), val)
        elif action == 2: # Inversion
            i, j = sorted(random.sample(range(n), 2))
            new_order[i:j] = reversed(new_order[i:j])
        return new_order

    def work_with_collab(self, emp, mode="Amis"):
        # 1. Interaction Collaboration (Amis ou Ennemis)
        if emp.best_global_order:
            if mode == "Amis":
                self.current_order = list(emp.best_global_order)
            elif mode == "Ennemis" and emp.best_global_cmax < self.best_cmax:
                self.current_order = list(emp.best_global_order)

        # 2. Choix du voisinage (RL)
        action = np.argmax(self.q_table) if (self.use_rl and random.random() > self.epsilon) else random.randint(0, 2)
        old_val = self.best_cmax
        
        self.current_order = self.apply_neighborhood(self.best_order, action)
        self.work(iterations=10)
        
        # 3. Apprentissage
        if self.use_rl:
            reward = (old_val - self.best_cmax) if self.best_cmax < old_val else -0.1
            self.q_table[action] += self.alpha_rl * (reward - self.q_table[action])
        
        # 4. Partage
        emp.update(self.best_order, self.best_cmax)

# =================================================================
# 5. GÉNÉRATION DES DEUX TABLEAUX
# =================================================================
# =================================================================
# 5. GÉNÉRATION DES DEUX TABLEAUX (VERSION ABREGEE ET LISIBLE)
# =================================================================
if __name__ == "__main__":
    scenarios = [("J1", 10), ("J2", 20), ("J3", 30)]
    
    # --- TABLEAU 1 : SANS COLLABORATION ---
    print("\n" + "="*60 + "\nT1: SANS COLLABORATION (P.25)\n" + "="*60)
    res_solo = []
    for day, nb in scenarios:
        ops = generate_hospital_data(nb)
        for algo in ["AG", "TABOU", "RS"]:
            s = MetaHeuristicAgent("S", algo, list(range(1, nb+1)), ops).work(800)
            a = MetaHeuristicAgent("A", algo, list(range(1, nb+1)), ops)
            for _ in range(5): a.work(50)
            rl = CollabRLAgent("RL", algo, list(range(1, nb+1)), ops, use_rl=True)
            for _ in range(10): rl.work(50)
            res_solo.append({"Jr": day, "NbP": nb, "Alg": algo, "Solo": s, "Agent": a.best_cmax, "RL": rl.best_cmax})
    print(pd.DataFrame(res_solo).to_string(index=False))

    # --- TABLEAU 2 : AVEC COLLABORATION (SMA) ---
    print("\n" + "="*60 + "\nT2: AVEC COLLABORATION (SMA - P.26)\n" + "="*60)
    groupes = ["AG_T", "AG_RS", "T_RS", "ALL"]
    res_collab = []
    for day, nb in scenarios:
        ops = generate_hospital_data(nb)
        for learning in [False, True]:
            for mode in ["Amis", "Enn"]:
                for g in groupes:
                    emp = SharedMemory()
                    # Mapping des noms de groupes vers les algos
                    mapping = {"AG_T": ["AG", "TABOU"], "AG_RS": ["AG", "RS"], 
                               "T_RS": ["TABOU", "RS"], "ALL": ["AG", "TABOU", "RS"]}
                    agents = [CollabRLAgent(name, name, list(range(1, nb+1)), ops, use_rl=learning) 
                              for name in mapping[g]]
                    
                    for _ in range(10): 
                        for a in agents: a.work_with_collab(emp, mode="Amis" if mode=="Amis" else "Ennemis")
                    
                    suffix = "RL" if learning else "NoRL"
                    res_collab.append({
                        "Scenario": f"{day}({nb}p)",
                        "Config": f"{mode}_{suffix}",
                        "Groupe": g,
                        "Cmax": emp.best_global_cmax
                    })

    # On transforme le tableau pour qu'il soit lisible : 
    # Scénarios en colonnes, Configurations en lignes
    df_collab = pd.DataFrame(res_collab)
    pivot_df = df_collab.pivot_table(index=['Config', 'Groupe'], columns='Scenario', values='Cmax')
    
    print(pivot_df)
    print("\nLégende: ALL=AG+Tabou+RS, T=Tabou, Enn=Ennemis")