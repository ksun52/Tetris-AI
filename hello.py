import numpy as np
from collections import defaultdict
import math
import copy
import random
from tetris_game import Tetris

class TetrisNode:
    def __init__(self, game_state=None, parent=None, piece=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.piece = piece  # The actual piece ID
        self.action = action  # (x, rotation)
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_pieces = list(range(7))  # All possible pieces
        self.untried_actions = []  # Will be populated when piece is selected
        
    def add_child(self, piece, action=None, game_state=None):
        child = TetrisNode(
            game_state=game_state,
            parent=self,
            piece=piece,
            action=action
        )
        key = (piece, action) if action else piece
        self.children[key] = child
        return child

    def get_ucb(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)

class TetrisMCTS:
    def __init__(self, simulation_count=150):
        self.simulation_count = simulation_count
        
    def get_next_move(self, game):
        root = TetrisNode(game_state=copy.deepcopy(game))
        
        for _ in range(self.simulation_count):
            node = root
            game_state = copy.deepcopy(game)
            
            # Selection
            while node.untried_pieces == [] and node.untried_actions == [] and node.children:
                node = self._select_child(node)
                if node.action:  # If this is a move node, apply the move
                    game_state.play(node.action[0], node.action[1])
            
            # Expansion
            if node.untried_pieces:
                # Expand by trying a new piece
                piece = random.choice(node.untried_pieces)
                node.untried_pieces.remove(piece)
                
                # Create new game state with this piece
                new_state = copy.deepcopy(game_state)
                new_state.current_piece = piece
                child = node.add_child(piece=piece, game_state=new_state)
                
                # Calculate possible actions for this piece
                child.untried_actions = list(new_state.get_next_states().keys())
                node = child
                game_state = new_state
                
            elif node.untried_actions:
                # Expand by trying a new action for the current piece
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                new_state = copy.deepcopy(game_state)
                new_state.play(action[0], action[1])
                child = node.add_child(
                    piece=node.piece,
                    action=action,
                    game_state=new_state
                )
                node = child
                game_state = new_state
            
            # Simulation
            score = self._simulate(game_state)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.value += score
                node = node.parent
        
        # Choose best move
        best_value = float('-inf')
        best_action = None
        
        for child_key, child in root.children.items():
            if isinstance(child_key, tuple) and child.visits > 0:  # Only consider move nodes
                value = child.value / child.visits
                if value > best_value:
                    best_value = value
                    best_action = child_key[1]  # The action part of the tuple
                    
        return best_action
    
    def _select_child(self, node):
        return max(node.children.values(), key=lambda child: child.get_ucb())
    
    def _simulate(self, game_state):
        simulation_game = copy.deepcopy(game_state)
        depth = 0
        max_depth = 10
        total_score = 0
        
        while not simulation_game.game_over and depth < max_depth:
            # Get possible moves for current piece
            possible_moves = list(simulation_game.get_next_states().keys())
            if not possible_moves:
                break
                
            # Choose move using simple heuristic
            best_move = None
            best_score = float('-inf')
            
            for move in possible_moves:
                test_game = copy.deepcopy(simulation_game)
                score, _ = test_game.play(move[0], move[1])
                
                # Calculate heuristic score
                state_props = test_game._get_board_props(test_game.board)
                heuristic = (
                    score * 1.0 +  # Actual score
                    -0.5 * state_props[1] +  # Holes penalty
                    -0.1 * state_props[2] +  # Bumpiness penalty
                    -0.2 * state_props[3]    # Height penalty
                )
                
                if heuristic > best_score:
                    best_score = heuristic
                    best_move = move
            
            # Apply best move
            score, game_over = simulation_game.play(best_move[0], best_move[1])
            total_score += score
            depth += 1
        
        return total_score

class TetrisAI:
    def __init__(self, render=True):
        self.game = Tetris()
        self.mcts = TetrisMCTS(simulation_count=150)
        self.render = render

    def play_game(self):
        total_score = 0
        moves = 0
        
        while not self.game.game_over:
            # Get best move from MCTS
            move = self.mcts.get_next_move(self.game)
            
            # Make the move
            score, game_over = self.game.play(move[0], move[1], 
                                            render=self.render, 
                                            render_delay=0.05)
            total_score += score
            moves += 1
            
            if moves % 10 == 0:
                print(f"Moves: {moves}, Score: {total_score}")
        
        print(f"Game Over! Final Score: {total_score}, Moves: {moves}")
        return total_score

if __name__ == "__main__":
    ai = TetrisAI(render=True)
    ai.play_game()