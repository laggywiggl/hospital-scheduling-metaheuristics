# Optimisation de la Planification Hospitaliere - Metaheuristiques

Projet universitaire d'**Aide a la Decision** explorant differentes metaheuristiques pour resoudre un probleme de planification d'operations medicales sous contraintes.

## Problematique

Planifier de maniere optimale un ensemble d'operations medicales en respectant les contraintes reelles (sequence d'operations, ressources limitees, competences requises) afin de **minimiser le temps total de traitement (makespan)**.

## Algorithmes Implementes

| Fichier | Methode | Auteurs |
|---------|---------|---------|
| `Recherche Tabou.ipynb` | Recherche Tabou | Ouafa AIT OUAMER, Hassan MAHAMAT IMAM |
| `Algo_genetique.ipynb` | Algorithme Genetique | Imad FECIH, Oussama AMIRI |
| `Algorithme génétique 2.ipynb` | Algorithme Genetique (variante) | — |
| `Projet_IA_optimisation_RS.ipynb` | Recuit Simule | Bakir KOUTA, Timote RICHARD |
| `SMA_orchestration.py` | Systeme Multi-Agents (orchestration) | — |
| `agent.py` | Agent SMA individuel | — |

## Structure du Projet

```
.
├── Recherche Tabou.ipynb          # Tabu Search
├── Algo_genetique.ipynb           # Genetic Algorithm
├── Algorithme génétique 2.ipynb   # Genetic Algorithm (variant)
├── Projet_IA_optimisation_RS.ipynb# Simulated Annealing
├── SMA_orchestration.py           # Multi-Agent System orchestrator
├── agent.py                       # Individual agent logic
├── test.py                        # Test script
├── test_Imad.py                   # Parametrised experiment driver
├── test_ouassa.ipynb              # Combined test notebook
├── requirements.txt               # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
```

## Utilisation

### Notebooks Jupyter

```bash
jupyter notebook
```

Ouvrir le notebook correspondant a l'algorithme souhaite et executer les cellules.

### Scripts Python

```bash
python SMA_orchestration.py
```

## Dependances

- Python 3.9+
- NumPy
- Pandas
- Matplotlib

## Licence

Projet academique — Master IA, Aide a la Decision.
