import numpy as np
from collections import defaultdict
import math
import copy
import random
from tetris_game import Tetris

from constants import SIM_COUNT, MAX_DEPTH, UCB_C


class TetrisNode:
    def __init__(self, parent, piece, pos, rotation):
        self.parent = parent
        self.children = []
        self.possible_children = [] # list of tuples (piece, position, rotation)
        self.level = None
        
        self.piece = piece
        self.action = tuple(pos, rotation)
        # self.bag = bag
        
        self.num_playouts = 0
        self.total_reward = 0

        if parent:
            self.level = 1
        else:
            self.level = parent.level + 1
        

    def update():
        pass


class TetrisMCTS:
    '''
    Generates the MCTS simulation that selects best moves 
    '''
    
    def __init__(self, max_simulations=SIM_COUNT, max_playout_depth=MAX_DEPTH, game=None):
        self.simulation_count = max_simulations
        self.playout_depth = max_playout_depth
        
        self.game = game
        self.MCTS_game_state = None # copy of self.game that is generated on each get_best_move simulation

        # start off the MCTS root node
        self.root = TetrisNode(None, self.game.current_piece, self.game.current_pos, self.game.current_rotation)
        root_next_states = self.game.get_next_states()
        root_next_actions = root_next_states.keys()
        self.root.possible_children = [TetrisNode(self, self.game.current_piece, action[0], action[1]) for action in root_next_actions]
    
    def get_best_move(self, root):
        '''
        MCTS ALGORITHM
        Issues a new game state copy for every simulation iteration -- allows us to use game's helper funcs 
        Every iteration is a new "game" that we must track 
        '''

        # repeat this for max_simulations count
        for _ in range(self.simulation_count):
            # save a new state of the game from here -- update it as you traverse through in select 
            self.MCTS_game_state = copy(self.game)
            
            # want to ensure that we're not learning conditioned to a fixed bag i.e. learning the bag itself
            # shuffle the game's bag for this simulation run 
            self.MCTS_game_state.bag.append(self.MCTS_game_state.next_piece)
            random.shuffle(self.MCTS_game_state.bag)
            self.MCTS_game_state.next_piece = self.MCTS_game_state.bag.pop()

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

    def _select(self, node):
        ''' 
        Starts at the root and traverses the tree until a node with possible children that aren't visited are found 
        If the node has visited all of its possible children, use UCB to traverse 
        "possible children" must also be the children that match the current piece from the game 
        Returns a node with "possible_children" and passes it to expand 
        
        TODO: what if it doesnt have any children

        '''
        if node == self.root:
            return node
        
        while node.possible_children and not self.MCTS_game_state.game_over:

            # select this node if it has possible children not explored
            possible_children_w_piece = [pc fo\r pc in node.possible_children if pc.piece == self.MCTS_game_state.current_piece]
            explored_children_w_piece = [c for c in node.children if c.piece == self.MCTS_game_state.current_piece]
            if len(explored_children_w_piece) < len(possible_children_w_piece):
                self.MCTS_game_state.play(node.action[0], node.action[1])
                return node

            # otherwise use UCB to traverse down -- UCB over the possible children that match the current piece 
            node_temp = self._ucb(explored_children_w_piece)
            
            # implement using tetris_game.py into selection - new game state for each run through in get_best_move loop
            # after selecting the next node to go down, play a turn using the Tetris game 
            self.MCTS_game_state.play(node.action[0], node.action[1])
            node = node_temp
        return node


    def _gen_children(self, piece_id = None):
        '''Get all possible next stah ates for a node'''
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
                while not self.MCTS_game_state._check_collision(piece_composition, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self.MCTS_game_state._add_piece_to_board(piece_composition, pos)
                    states[(piece_id, x, rotation)] = self.MCTS_game_state._get_board_props(board)

        return states


    def expansion_policy(self, node):
        # from nodes in unvisited select a random PIECE TYPE and then use expansion policy to select translation/rotation from one of those nodes
        # the random PIECE TYPE is already selected from the game's bag 
        
        # possible_pieces = set([node.piece for node in unvisited])
        # random_piece = random.choice(possible_pieces)
        # updated_unvisited = [node for node in unvisited if node.piece == random_piece]
        

        # random expansion for now, could choose with model/heuristic when available
        possible_children_w_piece = [pc for pc in node.possible_children if pc.piece == self.MCTS_game_state.current_piece]
        explored_children_w_piece = [c for c in node.children if c.piece == self.MCTS_game_state.current_piece]
        unvisited = set(possible_children_w_piece) - set(explored_children_w_piece)
        return random.choice(unvisited)


    def _expand(self, node):
        '''
        Takes in a node that has unvisited "possible_children"
        Selects an unvisited "possible child" according to the current piece and expansion policy (random or otherwise) and creates a node for it
        Generates the "possible_children" set for this new child node 
        Returns the child node and passes it to simulate 

        TODO: what if node doesnt have any children
        ''' 
        
        if not node.possible_children or self.MCTS_game_state.game_over:
            return node

        # selects a successor node to play through from the node's unexplored possible children 
        play_through = self.expansion_policy(node)
        self.MCTS_game_state.play(node.action[0], node.action[1])

        # generate "possible_children" set for the new child node 
        tetris_game_states = {}
        for piece_id in range(7):   # technically we should only do this for the pieces still in bag but its ok to have redundant nodes
            tetris_game_states.update(self._gen_children(play_through, piece_id))

        piece_states = tetris_game_states.keys()
        play_through.possible_children = [TetrisNode(play_through, state[0], state[1], state[2]) for state in piece_states]
        
        # append play_through node to the current node and return play_through 
        node.children.append(play_through)
        return play_through

    
    def get_next_piece(self, bag):
        if len(bag) == 0:
            bag = random.shuffle(range(7))
        return bag.pop()
    
    
    def _simulate(self, node):
        '''
        while game not done and depth < max depth
            choose next node from children via heuristcs
            generate next piece from bag (if bag empty repopulate it)
            define children for node based on next piece
            node = next node
        return sc
        '''
        game_copy = copy(self.MCTS_game_state)
        done = False
        depth = 0
        bag = copy(node.bag)

        while not done and depth < self.playout_depth:
            next_piece = self.get_next_piece(bag)
            actions = self._gen_children(node, next_piece).keys()
            
            chosen_action = random.choice(actions)
            score, game_over
            # node = TetrisNode(node, chosen_action[0], chosen_action[1], chosen_action[2], next_bag)
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
        self.mcts = TetrisMCTS(simulation_count=SIM_COUNT, max_playout_depth=MAX_DEPTH, game=self.game)

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