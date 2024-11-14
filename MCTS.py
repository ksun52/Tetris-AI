import numpy as np
from collections import defaultdict
import math
import copy
import random
from tetris_game import Tetris

UCB_C = np.sqrt(2)


class TetrisNode:
    def __init__(self, parent, piece, translation, rotation, bag):
        self.parent = parent
        self.children = []
        self.possible_children = []
        self.level = None
        
        self.piece = piece
        self.action = tuple(translation, rotation)
        self.bag = bag
        
        self.num_playouts = 0
        self.total_reward = 0
        

    def update():
        pass


class TetrisMCTS:
    def __init__(self, max_simulations = 150, max_playout_depth = 20, game = None):
        self.simulation_count = max_simulations
        self.playout_depth = max_playout_depth
        self.game = game
        self.MCTS_game_state = None
    
    # TODO: issue a new game state for every loop of get best move
    # every MCTS simulation is a "game" and we need to use all of the tetris game funcs
    def get_best_move(self, root):
        '''MCTS ALGORITHM'''

        # repeat this for max_simulations count
        for _ in range(self.simulation_count):
            # save a new state of the game from here -- update it as you traverse through in select 
            self.MCTS_game_state = copy(self.game)

            to_expand = self._select(root)
            to_play_through = self._expand(to_expand)
            end_state_value = self._simulate(to_play_through)
            self._backpropagate(to_play_through, end_state_value)
        # return most visited node

        #TODO: update the actual self.game
        return 

    def _ucb(self, nodes):
        # decide which node to select via ucb-1 node
        max_score = float('-inf')
        best_node = nodes[0]
        for node in nodes:
            exploitation = node.total_reward / node.num_playouts
            exploration = np.sqrt(np.log(node.parent.num_playouts) / node.num_playouts)
            score = exploitation + UCB_C * exploration
            if score > max_score:
                max_score = score
                best_node = node
        return best_node
    
    

    '''
    Starts at the root and traverses the tree until a node with possible children that aren't visited are found 
    If the node has visited all of its possible children, use UCB to traverse 
    Returns a node with "possible_children" and passes it to expand 
    '''
    def _select(self, node):
        while True:
            if len(node.children) < len(node.possible_children):
                return node
            node_temp = self._ucb(node.children)
            # implement using tetris_game.py into selection - new game state for each run through in get_best_move loop
            self.MCTS_game_state.play()
            node = node_temp

        

    '''Get all possible next states for a node'''
    def _gen_children(self, node, piece_id = None):
        states = {}
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece_composition = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece_composition])
            max_x = max([p[0] for p in piece_composition])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self.game._check_collision(piece_composition, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self.game._add_piece_to_board(piece_composition, pos)
                    states[(piece_id, x, rotation)] = self.game._get_board_props(board)

        return states

    def expansion_policy(unvisited):
        # from nodes in unvisited select a random PIECE TYPE and then use expansion policy to select translation/rotation from one of those nodes
        possible_pieces = set([node.piece for node in unvisited])
        random_piece = random.choice(possible_pieces)
        updated_unvisited = [node for node in unvisited if node.piece == random_piece]
        # random expansion for now, could choose with model/heuristic when available
        return random.choice(updated_unvisited)
    
    '''
    Takes in a node that has unvisited "possible_children"
    Selects an unvisited "possible child" according to expansion policy (random or otherwise) and creates a node for it
    Generates the "possible_children" set for this new child node 
    Returns the child node and passes it to simulate 
    '''
    def _expand(self, node):
        unvisited = set(node.possible_children) - set(node.children)
        play_through = self.expansion_policy(unvisited)
        
        tetris_game_states = {}
        for piece_id in range(7):
            tetris_game_states.update(self._gen_children(play_through, piece_id))

        piece_states = tetris_game_states.keys()
        
        
        if len(node.bag == 1):
            next_bag = random.shuffle(range(7))
        else:
            next_bag = node.bag[:-1]
        
        play_through.possible_children = [TetrisNode(node, state[0], state[1], state[2], next_bag) for state in piece_states]
        node.children.append(play_through)
        return play_through

    
    def get_next_piece(self, bag):
        if len(bag) == 0:
            bag = random.shuffle(range(7))
        return bag.pop()
    
    '''
    while game not done and depth < max depth
        choose next node from children via heuristcs
        generate next piece from bag (if bag empty repopulate it)
        define children for node based on next piece
        node = next node
    return sc
    '''
    def _simulate(self, node):
        game_copy = copy(self.game)
        done = False
        depth = 0
        bag = copy(node.bag)

        while not done and depth < self.playout_depth:
            next_piece = self.get_next_piece(bag)
            actions = self._gen_children(node, next_piece).keys()
            
            chosen_action = random.choice(actions)
            score, game_over
            node = TetrisNode(node, chosen_action[0], chosen_action[1], chosen_action[2], next_bag)
            # check if we lost
            game_copy.play()
        
        pass

    def _backpropagate(self, node, reward):
        while node.parent != None:
            node.total_reward += reward * 1.0
            node.num_playouts += 1.0
            node = node.parent
        
        # backprop on root node
        node.total_reward += reward
        node.num_playouts += 1.0


class TetrisAI:
    def __init__(self, render=True):
        self.game = Tetris()
        self.mcts = TetrisMCTS(simulation_count = 150, game = self.game)

    def play_game(self):
        total_score = 0
        moves = 0

        # TODO think hard about logic for starting off the MCTS with root node 

        while not self.game.game_over:
            # find the best move from MCTS 
            self.mcts.get_best_move(self.game)
            
            # play the game 
            
            # update the score 





if __name__ == "__main__":
    ai = TetrisAI(render=True)
    ai.play_game()