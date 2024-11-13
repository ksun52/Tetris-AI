import numpy as np
from collections import defaultdict
import math
import copy
from tetris_game import Tetris

UCB_C = np.sqrt(2)

class TetrisNode:
    def __init__(self, parent, piece, translation, rotation):
        self.parent = parent
        self.children = []
        self.level = None
        
        self.piece = piece
        self.action = tuple(translation, rotation)
        
        self.num_playouts = 0
        self.total_reward = 0
        

    def update():
        pass


class TetrisMCTS:
    def __init__(self, max_simulations = 150):
        self.simulation_count = max_simulations

    
    def get_best_move(self, root):
        '''MCTS ALGORITHM'''

        # repeat this for max_simulations count
        
        # while still have simulations to play out
            # selection  - returns node to expand
            # expansion - returns node to play through
            # simulation - returns value of playout
            # backpropagation
        # return most visited node
        return 

    def _ucb(self, nodes):
        # assumes each node in nodes has been explored at least onc
        # decide which node to select via ucb-1 node
        max_score = float('i-inf') 
        max_node
        for
        pass
    
    def _select(self, node):
        # traverse down the tree
        # while node has children and are all fully expanded:
        while True:
            unvisited_children = [child for child in node.children if child.total_playouts == 0]
            if len(unvisited_children) > 0:
                return node.children[0]
                
            node = _ucb(node.children)


    def _expand(self, node):
        # generate all possible child nodes from selected node 
        # add all possible states from node to the tree as leaf nodes with 0/0 MCTS score
        
        # if root:
            # generate children based on rotation and translation
        #deeper states:
            # generate children based on piece, rotation, translation 
        pass

    def _simulate(self, node):
        # while game not done and depth < max depth
            # choose next node from children via heuristcs
            # generate next piece from bag (if bag empty repopulate it)
            # define children for node based on next piece
            # node = next node
        # return sc
        pass

    def _backpropagate(self, node, reward):
        while node.parent != None:
            node.total_reward += reward
            node.num_playouts += 1
            node = node.parent
        
        # backprop on root node
        node.total_reward += reward
        node.num_playouts += 1


class TetrisAI:
    def __init__(self, render=True):
        self.game = Tetris()
        self.mcts = TetrisMCTS(simulation_count = 150)

    def play_game(self):
        total_score = 0
        moves = 0
        while not self.game.game_over:
            # find the best move from MCTS 
            self.mcts.get_best_move(self.game)
            
            # play the game 
            
            # update the score 





if __name__ == "__main__":
    ai = TetrisAI(render=True)
    ai.play_game()