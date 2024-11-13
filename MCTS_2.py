import numpy as np
from collections import defaultdict
import math
import copy
from tetris_game import Tetris

class TetrisNode:
    def __init__(self, game_state=None, parent=None, is_piece_node=True, piece=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.is_piece_node = is_piece_node  # True if this node represents piece selection
        self.piece = piece  # The piece that was selected (for move nodes)
        self.action = action  # (x, rotation) tuple for move nodes
        self.children = {}
        self.visits = 0
        self.value = 0

    def add_child(self, key, game_state=None, is_piece_node=True, piece=None, action=None):
        child = TetrisNode(
            game_state=game_state,
            parent=self,
            is_piece_node=is_piece_node,
            piece=piece,
            action=action
        )
        self.children[key] = child
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def get_ucb(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)

class TetrisMCTS:
    def __init__(self, simulation_count=100):
        self.simulation_count = simulation_count

    def get_best_move(self, game):
        root = TetrisNode(game_state=copy.deepcopy(game), is_piece_node=False)
        
        # Run simulations
        for _ in range(self.simulation_count):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.game_state.game_over:
                node = self._expand(node)
            
            # Simulation
            reward = self._simulate(node)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Choose best action based on average value
        best_action = max(root.children.items(),
                         key=lambda x: x[1].value / max(x[1].visits, 1))[0]
        
        return best_action

    def _select(self, node):
        while node.children and not node.game_state.game_over:
            if node.is_piece_node:
                # For piece nodes, check if all possible pieces have been tried
                if len(node.children) < 7:  # 7 different tetris pieces
                    return node
            else:
                # For move nodes, check if all possible moves for the current piece have been tried
                if len(node.children) < len(node.game_state.get_next_states()):
                    return node
            
            node = max(node.children.values(), key=lambda x: x.get_ucb())
        return node

    def _expand(self, node):
        if node.is_piece_node:
            # Expand piece node by selecting an untried piece
            available_pieces = set(range(7))  # 7 different tetris pieces
            tried_pieces = set(node.children.keys())
            untried_pieces = available_pieces - tried_pieces
            
            if untried_pieces:
                new_piece = untried_pieces.pop()
                # Create a new game state with this piece
                new_game = copy.deepcopy(node.game_state)
                new_game.current_piece = new_piece
                child = node.add_child(
                    key=new_piece,
                    game_state=new_game,
                    is_piece_node=False,
                    piece=new_piece
                )
                return child
        else:
            # Expand move node by trying an untried move for the current piece
            possible_states = node.game_state.get_next_states()
            unvisited_actions = set(possible_states.keys()) - set(node.children.keys())
            
            if unvisited_actions:
                action = unvisited_actions.pop()
                new_game = copy.deepcopy(node.game_state)
                new_game.play(action[0], action[1])
                child = node.add_child(
                    key=action,
                    game_state=new_game,
                    is_piece_node=True,
                    action=action
                )
                return child
                
        return node

    def _simulate(self, node):
        game = copy.deepcopy(node.game_state)
        total_reward = 0
        depth = 0
        max_depth = 10  # Limit simulation depth
        
        while not game.game_over and depth < max_depth:
            # First, randomly select next piece if we're at a piece selection point
            if len(game.bag) == 0:
                game.bag = list(range(len(Tetris.TETROMINOS)))
                np.random.shuffle(game.bag)
            
            # Then, choose a random valid move for the current piece
            possible_moves = list(game.get_next_states().keys())
            if not possible_moves:
                break
                
            x, rotation = possible_moves[np.random.randint(len(possible_moves))]
            reward, game_over = game.play(x, rotation)
            total_reward += reward
            depth += 1
            
        # Add heuristic evaluation of final state
        if not game.game_over:
            state_props = game._get_board_props(game.board)
            heuristic_value = (
                -0.5 * state_props[1] +  # Holes
                -0.1 * state_props[2] +  # Bumpiness
                -0.2 * state_props[3]    # Height
            )
            total_reward += heuristic_value
            
        return total_reward

    def _backpropagate(self, node, reward):
        while node:
            node.update(reward)
            node = node.parent

class TetrisAI:
    def __init__(self, render=True):
        self.game = Tetris()
        self.mcts = TetrisMCTS(simulation_count=150)  # Increased simulation count
        self.render = render

    def play_game(self):
        total_score = 0
        moves = 0
        
        while not self.game.game_over:
            # Get best move from MCTS
            x, rotation = self.mcts.get_best_move(self.game)
            
            # Make the move
            score, game_over = self.game.play(x, rotation, render=self.render, render_delay=0.05)
            total_score += score
            moves += 1
            
            if moves % 10 == 0:
                print(f"Moves: {moves}, Score: {total_score}")
        
        print(f"Game Over! Final Score: {total_score}, Moves: {moves}")
        return total_score

# Example usage
if __name__ == "__main__":
    ai = TetrisAI(render=True)
    ai.play_game()