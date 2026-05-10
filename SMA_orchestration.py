import random
import math
import heapq
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import re
from typing import Dict, List, Tuple

# Fix random seed for reproducibility
random.seed(42)

# -----------------------------
# Constants
# -----------------------------
tasks = ["C1", "C2", "C3", "C4", "C5", "C6"]
TASK_TIME = 1
SLOT_MINUTES = 1


# -----------------------------
# Generate Random Patient Data
# -----------------------------
def generate_random_patients(n_patients: int) -> List[List[List[str]]]:
    """
    Generate random patient operations.
    Each patient has 5 operations, each with 0-3 skills, each skill with multiplicity 1-3.
    """
    skills = ["C1", "C2", "C3", "C4", "C5", "C6"]
    patients = []
    for _ in range(n_patients):
        ops = []
        for _ in range(5):  # 5 operations per patient
            n_skills = random.randint(0, 3)  # 0 to 3 skills per operation
            op = []
            for _ in range(n_skills):
                skill = random.choice(skills)
                mult = random.randint(1, 3)
                if mult == 1:
                    op.append(skill)
                else:
                    op.append(f"{skill}*{mult}")
            ops.append(op)
        patients.append(ops)
    return patients


# -----------------------------
# Preprocessing (from Tabu)
# -----------------------------
def parse_task(token: str) -> Tuple[str, int]:
    m = re.fullmatch(r"(C\d+)(?:\*(\d+))?", token.strip())
    if not m:
        raise ValueError(f"Token de tâche invalide: {token}")
    skill = m.group(1)
    mult = int(m.group(2)) if m.group(2) else 1
    return skill, mult


def expand_patient_ops(p_ops: List[List[str]]) -> List[Dict[str, int]]:
    stages: List[Dict[str, int]] = []
    for col in p_ops:
        counts: Dict[str, int] = {}
        for tok in col:
            skill, k = parse_task(tok)
            counts[skill] = counts.get(skill, 0) + k
        stages.append(counts)
    return stages


def preprocess(patients_ops: List[List[List[str]]]) -> List[List[Dict[str, int]]]:
    return [expand_patient_ops(p) for p in patients_ops]


# patients_stages = preprocess(patients_operations)  # Removed, using generated data


# -----------------------------
# Simulate function (from Tabu)
# -----------------------------
def simulate(
    order: List[int],
    stages: List[List[Dict[str, int]]],
    capacities: Dict[str, int],
) -> Tuple[int, List[int]]:
    n = len(stages)
    remaining = [[dict(d) for d in stages[i]] for i in range(n)]
    next_stage = [0] * n
    finished = [False] * n
    finish_times = [0] * n

    t = 0
    done = 0
    while done < n:
        for i in range(n):
            if finished[i]:
                continue
            k = next_stage[i]
            while k < len(remaining[i]) and sum(remaining[i][k].values()) == 0:
                k += 1
            next_stage[i] = k
            if k >= len(remaining[i]):
                finished[i] = True
                finish_times[i] = t
                done += 1

        if done >= n:
            break

        avail = dict(capacities)
        progress = False

        for pid in order:
            if finished[pid]:
                continue
            k = next_stage[pid]
            if k >= len(remaining[pid]):
                continue
            stage = remaining[pid][k]
            for s in list(stage.keys()):
                need = stage[s]
                if need <= 0:
                    continue
                cap = avail.get(s, 0)
                if cap <= 0:
                    continue
                take = min(need, cap)
                stage[s] -= take
                avail[s] -= take
                if take > 0:
                    progress = True

        if not progress:
            raise RuntimeError("Blocage détecté: aucune progression dans ce tick.")

        t += 1

    return t, finish_times


# -----------------------------
# Tabu Search Heuristic
# -----------------------------
def tabu_search(
    stages: List[List[Dict[str, int]]],
    capacities: Dict[str, int],
    max_iter: int = 800,
    tabu_tenure: int = 7,
    stop_no_improve: int = 30,
    seed: int = 0,
) -> Tuple[int, List[int]]:
    random.seed(seed)
    n = len(stages)

    def workload(i: int) -> int:
        return sum(v for d in stages[i] for v in d.values())

    current = sorted(range(n), key=workload, reverse=True)

    cache: Dict[Tuple[int, ...], int] = {}

    def cost(perm: List[int]) -> int:
        key = tuple(perm)
        if key not in cache:
            cache[key] = simulate(perm, stages, capacities)[0]
        return cache[key]

    best = current[:]
    best_cost = cost(best)

    tabu: Dict[Tuple[int, int], int] = {}
    no_imp = 0

    for it in range(1, max_iter + 1):
        best_neighbor = None
        best_neighbor_cost = float("inf")
        best_move = None

        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = current[:]
                cand[i], cand[j] = cand[j], cand[i]
                move = tuple(sorted((current[i], current[j])))

                c = cost(cand)
                is_tabu = tabu.get(move, -1) >= it
                if is_tabu and c >= best_cost:
                    continue

                if c < best_neighbor_cost or (
                    c == best_neighbor_cost and random.random() < 0.5
                ):
                    best_neighbor = cand
                    best_neighbor_cost = c
                    best_move = move

        if best_neighbor is None:
            tabu.clear()
            continue

        current = best_neighbor
        current_cost = best_neighbor_cost

        tabu[best_move] = it + tabu_tenure

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= stop_no_improve:
                break

    return best_cost, best


# -----------------------------
# GA Heuristic (adapted from Algorithme génétique 2.ipynb)
# -----------------------------
DURATION_PER_TASK = 1
POPULATION_SIZE = 20
CROSSOVER_PROB = 0.075
MUTATE_SWAP_PROB = 0.05
MUTATE_INSERT_PROB = 0.05
GENERATIONS = 200


# Build tasks from patients_stages
def build_tasks_from_stages(stages):
    tasks_list = []
    tid = 0
    for pid, patient_stages in enumerate(stages):
        for op_idx, stage in enumerate(patient_stages):
            for skill, count in stage.items():
                for _ in range(count):
                    tasks_list.append(
                        {
                            "tid": tid,
                            "patient": pid + 1,
                            "op": op_idx + 1,
                            "skill": skill,
                            "dur": DURATION_PER_TASK,
                        }
                    )
                    tid += 1
    return tasks_list


# tasks_list = build_tasks_from_stages(patients_stages)  # Not used in new version
# N_TASKS = len(tasks_list)
# SKILLS = sorted({t["skill"] for t in tasks_list})

# # Predecessors
# predecessors = {}
# for t in tasks_list:
#     predecessors[t["tid"]] = set()

# For simplicity, assume no predecessors as in GA notebook, but actually there are.
# In GA, they build with predecessors based on ops.

# To simplify, since Tabu and RS don't use task order, perhaps adapt GA to patient order.

# For consistency, let's make GA also return patient order.

# But in GA notebook, it's task order.

# To make it simple, I'll adapt GA to work on patient permutations like Tabu.

# So, for GA, use the same simulate as Tabu.


def ga_heuristic(stages, capacities, pop_size=20, gens=200):
    n = len(stages)

    def fitness(perm):
        return simulate(perm, stages, capacities)[0]

    pop = [random.sample(range(n), n) for _ in range(pop_size)]
    for gen in range(gens):
        pop.sort(key=fitness)
        best = pop[0]
        new_pop = [best]  # elitism
        while len(new_pop) < pop_size:
            p1 = random.choice(pop[: pop_size // 2])
            p2 = random.choice(pop[: pop_size // 2])
            # crossover
            a, b = sorted(random.sample(range(n), 2))
            child = [None] * n
            child[a : b + 1] = p1[a : b + 1]
            k = 0
            for g in p2:
                if g not in child:
                    while child[k] is not None:
                        k += 1
                    child[k] = g
            # mutate
            if random.random() < 0.1:
                i, j = random.sample(range(n), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
    best_perm = min(pop, key=fitness)
    return fitness(best_perm), best_perm


# -----------------------------
# RS Heuristic (adapted from Projet_IA_optimisation_RS.ipynb)
# -----------------------------
def rs_on_order(stages, capacities, t0=1000, alpha=0.95, t_min=0.1, iter_cycle=100):
    n = len(stages)
    current = list(range(n))
    random.shuffle(current)
    best = current[:]
    best_cost = simulate(current, stages, capacities)[0]
    t = t0
    while t > t_min:
        for _ in range(iter_cycle):
            neighbor = current[:]
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            cost_n = simulate(neighbor, stages, capacities)[0]
            if cost_n < best_cost or random.random() < math.exp(
                (best_cost - cost_n) / t
            ):
                current = neighbor
                if cost_n < best_cost:
                    best = neighbor[:]
                    best_cost = cost_n
        t *= alpha
    return best_cost, best


# -----------------------------
# Agent Class
# -----------------------------
class Agent:
    def __init__(self, name, heuristic_func):
        self.name = name
        self.heuristic = heuristic_func

    def run(self, *args):
        return self.heuristic(*args)


# -----------------------------
# Orchestration
# -----------------------------
def main():
    agents = [
        ("AG", lambda stages, cap: ga_heuristic(stages, cap, pop_size=20, gens=200)),
        ("Tabou", lambda stages, cap: tabu_search(stages, cap, max_iter=800, seed=42)),
        ("RS", lambda stages, cap: rs_on_order(stages, cap)),
    ]

    patient_counts = [10, 20, 30]  # Test with 10, 20, 30 patients

    # Open file to write the table
    with open("resultats_tableau.md", "w", encoding="utf-8") as f:
        # Table header
        header1 = "| Nombre Patients | Métaheuristique | Agent_AG (Cmax) | Agent_Tabou (Cmax) | Agent_RS (Cmax) | Agent_AG_Apprentissage | Agent_Tabou_Apprentissage | Agent_RS_Apprentissage |\n"
        header2 = "|-----------------|-----------------|-----------------|-------------------|---------------|-------------------------|---------------------------|-----------------------|\n"
        f.write(header1)
        f.write(header2)
        print(header1.strip())
        print(header2.strip())

        for n_patients in patient_counts:
            # Generate random data
            patients_ops = generate_random_patients(n_patients)
            stages = preprocess(patients_ops)
            capacities = {t: 1 for t in tasks}  # Assuming 1 per skill, can adjust

            results = {}
            for name, func in agents:
                print(f"Running {name} for {n_patients} patients...")
                makespan, _ = func(stages, capacities)
                results[name] = makespan

            # For learning variants, placeholder (not implemented)
            ag_learning = "N/A"  # To be implemented
            tabou_learning = "N/A"
            rs_learning = "N/A"

            # Output row (one per patient count, with metaheuristics as sub)
            for meta in ["AG", "Tabou", "RS"]:
                ag_cmax = results.get("AG", "N/A")
                tabou_cmax = results.get("Tabou", "N/A")
                rs_cmax = results.get("RS", "N/A")
                row = f"| {n_patients} | {meta} | {ag_cmax} | {tabou_cmax} | {rs_cmax} | {ag_learning} | {tabou_learning} | {rs_learning} |\n"
                f.write(row)
                print(row.strip())

    print("Tableau sauvegardé dans 'resultats_tableau.md'")
    # Note: No collaboration, agents run independently.


if __name__ == "__main__":
    main()
