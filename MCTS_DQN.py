import numpy as np
import tensorflow as tf
from collections import deque
import random
import copy
import random
from tetris_game import Tetris
from constants import SIM_COUNT, MAX_DEPTH, UCB_C
from MCTS import TetrisMCTS
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense



class DQNAgent:
    def __init__(self, state_size=4, memory_size=10000, batch_size=32, modelFile=None):
        self.state_size = state_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.model = self._build_model()
        # self.target_model = self._build_model()
        self.update_target_counter = 0

        # load an existing model
        if modelFile is not None:
            self.model = load_model(modelFile)
        # create a new model
        else:
            self.model = self._build_model()


    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model


    def remember(self, state, reward, next_state, done):
        self.memory.append((state, reward, next_state, done))

    def get_heuristic_value(self, state):
        return self.model.predict(np.array([state]), verbose=0)[0][0]

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        rewards = np.array([transition[1] for transition in minibatch])
        next_states = np.array([transition[2] for transition in minibatch])
        dones = np.array([transition[3] for transition in minibatch])

        targets = rewards + self.gamma * self.model.predict(next_states, verbose=0)[:, 0] * (1 - dones)
        
        self.model.fit(states, targets, epochs=3, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
class TetrisMCTSWithDQN(TetrisMCTS):
    def __init__(self, simulation_count=SIM_COUNT, dqn_agent=None, game=None):
        super().__init__(max_simulations=simulation_count, game=game)
        self.dqn_agent = dqn_agent if dqn_agent else DQNAgent()


    def _simulate(self, expanded_node=None):
        state_props = self.mcts_game._get_board_props(self.mcts_game.board)
        predicted_value = self.dqn_agent.get_heuristic_value(state_props)
        return predicted_value


class TetrisAIWithDQN:
    def __init__(self, render=True):
        self.game = Tetris()
        self.dqn_agent = DQNAgent( modelFile=sys.argv[1])
        self.mcts = TetrisMCTSWithDQN(simulation_count=SIM_COUNT, dqn_agent=self.dqn_agent, game=self.game)
        self.render = render
        self.tetris_count = 0

    def train(self, num_episodes=2500):
        scores = []
        best_score = 0
        for episode in range(num_episodes):
            print(episode)
            self.game = Tetris()
            self.mcts = TetrisMCTSWithDQN(simulation_count=SIM_COUNT, dqn_agent=self.dqn_agent, game=self.game)
            
            total_score = 0
            moves = 0
            
            while not self.game.game_over:
                self.mcts = TetrisMCTSWithDQN(simulation_count=SIM_COUNT, dqn_agent=self.dqn_agent, game=self.game)
                # print(moves)
                move = self.mcts.get_best_move()
                old_state = self.game._get_board_props(self.game.board)
                
                score, game_over, lines_cleared = self.game.play(move.pos, move.rotation, 
                                                render=self.render and episode % 100 == 0, 
                                                render_delay=0.05 if episode % 100 == 0 else 0)
                
                new_state = self.game._get_board_props(self.game.board)
                self.dqn_agent.remember(old_state, score, new_state, game_over)
                
                total_score += score
                moves += 1
                
            
            scores.append(total_score)
            if episode % 1 == 0:
                self.dqn_agent.train()
                avg_score = np.mean(scores[-100:] if len(scores) >= 100 else scores)
                print(f"Episode: {episode}, Score: {total_score}, Average Score: {avg_score:.2f}")
            
            
            # Save the trained model
            if total_score >= best_score:
                best_score = total_score
                self.dqn_agent.model.save('tetris_dqn_model.keras')
        return scores

if __name__ == "__main__":
    ai = TetrisAIWithDQN(render=True)
    training_scores = ai.train(num_episodes=1000)
    
    # Plot training progress
    import matplotlib.pyplot as plt
    plt.plot(training_scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()