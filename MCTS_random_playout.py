import numpy as np
from collections import defaultdict
import math
import copy
import random
from tetris_game import Tetris
from keras.models import Sequential, load_model
from keras.layers import Dense
import sys

from constants import SIM_COUNT, MAX_DEPTH, UCB_C


class TetrisNode:
    def __init__(self, parent, piece, pos, rotation):
        self.parent = parent
        self.visited_children = []
        self.all_children = []

        self.piece = piece
        self.pos = pos
        self.rotation = rotation

        self.num_playouts = 0
        self.total_reward = 0


class TetrisMCTS:
    ''' Generates the MCTS simulation that selects best moves '''

    def __init__(self, max_simulations=SIM_COUNT, max_playout_depth=MAX_DEPTH, game=None):
        self.simulation_count = max_simulations
        self.playout_depth = max_playout_depth

        self.game = game
        self.mcts_game = None

        self.root = TetrisNode(None, None, None, None)
        root_children = self.game.get_next_states().keys()
        self.root.all_children = [TetrisNode(self.root, self.game.current_piece, action[0], action[1]) for action in root_children]

    def count_nodes(self, node):
        # Base case: if the node is None, return 0
        if node is None:
            return 0
        
        # Start with 1 to count the current node
        count = 1
        
        # Recursively count the children nodes
        for child in node.visited_children:
            count += self.count_nodes(child)
        
        return count


    def get_best_move(self):
        ''' MCTS ALGORITHM
        Issues a new game state copy for every simulation iteration -- allows us to use game's helper funcs 
        Every iteration is a new "game" that we must track 
        '''

        print("--------------------------------")
        for i in range(self.simulation_count):
            self.mcts_game = copy.deepcopy(self.game)
            
            selected_node = self._select(self.root)

            expanded_node = self._expand(selected_node)
            
            expanded_node_score = self._simulate(expanded_node)

            self._backpropagate(expanded_node, expanded_node_score)
        children_valu = [child.total_reward/child.num_playouts for child in self.root.visited_children]
        print(children_valu)
        print(self.count_nodes(self.root))
        for child in self.root.all_children:
            print(self.count_nodes(child))
        return max(self.root.visited_children, key=lambda child: child.total_reward/child.num_playouts)


    def _gen_children(self, piece_id = None):
        '''Get all possible next states for a node
        returns
            dict{
                key: (piece_id, position, rotation)
                value: board properties
            }
        '''
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
                while not self.mcts_game._check_collision(piece_composition, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self.mcts_game._add_piece_to_board(piece_composition, pos)
                    states[(piece_id, x, rotation)] = self.mcts_game._get_board_props(board)
        return states


    def _ucb(self, nodes):
        max_score = float('-inf')
        best_node = nodes[0]
        for node in nodes:
            exploitation = node.num_playouts
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
        '''
        # If the node does not have any children return this node.
        while len(node.all_children) != 0 and not self.mcts_game.game_over:

            # If the node has not visited all children, select this node
            if len(node.visited_children) < len(node.all_children):
                return node

            selected_node = self._ucb(node.visited_children)

            if self.mcts_game.current_piece != selected_node.piece:
                if self.mcts_game.next_piece == selected_node.piece:
                    self.mcts_game.next_piece = self.mcts_game.current_piece
                else:
                    self.mcts_game.bag.remove(selected_node.piece)
                    self.mcts_game.bag.append(self.mcts_game.current_piece)
                self.mcts_game.current_piece = selected_node.piece

            self.mcts_game.play(selected_node.pos, selected_node.rotation)
            node = selected_node

        return node


    def _expansion_policy(self, node):
        unvisited = set(node.all_children) - set(node.visited_children)
        return random.choice(list(unvisited))


    def _expand(self, selected_node):
        # If node does not have any children, return the selected node
        if len(selected_node.all_children) == 0 or self.mcts_game.game_over:
            return selected_node
        
        expanded_node = self._expansion_policy(selected_node)

        if self.mcts_game.current_piece != expanded_node.piece:
            
            if self.mcts_game.next_piece == expanded_node.piece:
                self.mcts_game.next_piece = self.mcts_game.current_piece
            else:
                self.mcts_game.bag.remove(expanded_node.piece)
                self.mcts_game.bag.append(self.mcts_game.current_piece)
            self.mcts_game.current_piece = expanded_node.piece
        self.mcts_game.play(expanded_node.pos, expanded_node.rotation)

        # Create children for the expanded nodes for future selection
        children = {}
        possible_pieces = copy.deepcopy(self.mcts_game.bag)
        possible_pieces.append(self.mcts_game.next_piece)
        for piece_id in possible_pieces:
            children.update(self._gen_children(piece_id))
        piece_states = children.keys()
        expanded_node.all_children = [TetrisNode(selected_node, move[0], move[1], move[2]) for move in piece_states]

        selected_node.visited_children.append(expanded_node)
        return expanded_node


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]


    def _playout_policy(self):
        next_states = {tuple(v):k for k, v in self.mcts_game.get_next_states().items()}
        states = next_states.keys()

        max_value = None
        best_state = None
        self.state_size=4

        for state in states:
            value = self.predict_value(np.reshape(state, [1, self.state_size]))
            if not max_value or value > max_value:
                max_value = value
                best_state = state
        best_action = next_states[best_state]
        return best_action


    def _simulate(self, expanded_node):
        if len(expanded_node.all_children) == 0 or self.mcts_game.game_over:
            return self.mcts_game.score

        depth = 0
        while  self.mcts_game.game_over and depth < self.playout_depth:
            pos, rotation = self._playout_policy()
            self.mcts_game.play(pos, rotation)
            depth += 1
        return self.mcts_game.score


    def _backpropagate(self, node, score):
        state_props = self.mcts_game._get_board_props(self.mcts_game.board)
        heuristic_value = (
            -0.4 * state_props[1] +  # Holes
            -0.1 * state_props[2] +  # Bumpiness
            -0.3 * state_props[3]    # Height
        ) / -10.0


        node.total_reward = score * 1.0
        node.num_playouts = 1.0
        while node.parent != None:
            node = node.parent
            node.total_reward += max(score, node.total_reward) * 1.0
            node.num_playouts = 1.0


    def make_move(self, best_move):
        self.game.play(best_move.pos, best_move.rotation, render=True, render_delay=0.01)
        best_move.parent = None
        self.root = best_move


class TetrisAI:
    def __init__(self, render=True, modelFile=sys.argv[1]):
        self.game = Tetris()
        self.mcts = TetrisMCTS(max_simulations=SIM_COUNT, max_playout_depth=MAX_DEPTH, game=self.game, modelFile=sys.argv[1])


    def play_game(self):
        moves = 0

        while not self.game.game_over:
            self.mcts = TetrisMCTS(max_simulations=SIM_COUNT, max_playout_depth=MAX_DEPTH, game=self.game, modelFile=sys.argv[1])
            best_move = self.mcts.get_best_move()

            self.mcts.make_move(best_move)
            
            moves += 1
        
        print("moves made" + str(self.game.num_moves))
        print("tetris made" + str(self.game.num_tetris))
        return self.game.score


if __name__ == "__main__":
    ai = TetrisAI(render=True)
    score = ai.play_game()
    print(score)
    # print(moves)