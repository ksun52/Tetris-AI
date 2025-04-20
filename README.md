# Overview

Tetris-AI is a Python‑based AI that plays Tetris with a human‑style objective: clear as many four‑line combos (“Tetrises”) as possible, rather than merely surviving. The code explores several agent designs and shows that a lightweight Mixture‑of‑Experts (MoE) approach triples the Tetris rate of a strong Deep Q‑Learning (DQL) baseline.

The project tests out
- Monte Carlo Tree Search (MCTS)
  - Baseline planner that plays out hundreds of future scenarios per move
- Monte Carlo Tree Search + Deep Q Learning (DQL)
  - Replaces random playouts with a pre-trained DQL evaluator  
- Mixture of Deep Q Learning experts
  - Two specialized DQL models plus a lightweight router that chooses which expert has the better move for each board state

# Code
`run_model.py`
This file is to run the Deep Q Learning model and MoE model one time and produce Tetris and score graphs.  
run it by executing:
```python3 run_model.py best_regular.keras best_small.keras```

`MCTS.py`
The MCTS that has a playout policy with the original heurisitcs based on the Genetic Algorithm Tetris paper.

`Heurisitic_search.py` 
This file was used to find the coefficients for the heurisitics used in MCTS.

`MCTS_heurisitc.py`
This MCTS has heurisitcs that we added to the base MCTS.py algorithm.

`MCTS_random_playout.py`
This file is the MCTS that has a random playout policy.

`MTCS_DQN.py`
This file is for MCTS with a Deep Q Learning playout policy.

# Contributors
This repo contains the work of Oskar Shiomi, Grant Murphy, and Kevin Sun
