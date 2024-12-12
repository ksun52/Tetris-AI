This repo contains the CSE 592 project of Oskar Shiomi, Grant Murphy, and Kevin Sun

run_model.py
this file is to run the Deep Q Learning model and MoE model one time and produce Tetris and score graphs.  
run it by executing:
```python3 run_model.py best_regular.keras best_small.keras```


MCTS.py  
the MCTS that has a playout policy with the original heurisitcs based on the Genetic Algorithm Tetris paper.

heurisitic_search.py  
this file was used to find the coefficients for the heurisitics used in MCTS.

MCTS_heurisitc.py  
this MCTS has heurisitcs that we added to the base MCTS.py algorithm.

MCTS_random_playout.py  
this files is the MCTS that has a random playout policy.

MTCS_DQN.py  
this file is for MCTS with a Deep Q Learning playout policy.
