import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    exit("Missing model file")

from dqn_agent import DQNAgent
from tetris_game_for_DQN import Tetris

env = Tetris()
regular_agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[1])
small_agent = DQNAgent(env.get_state_size(), modelFile=sys.argv[2])
done = False
agent = None

tetris = 0
almost_tetris = 0
moves = 0
score = 0
tetrises_adaptive = []
almost_tetrises_adaptive = []
scores_adaptive = []
agent = regular_agent
while not done and moves < 4000:
    max_height = env.get_max_height()
    min_height = env.get_min_height()
    holes_in_bottom_rows = env.has_holes_in_bottom_rows(rows=7)
    if max_height < 7 and not holes_in_bottom_rows and max_height-min_height < 4:
        # print("small")
        agent = small_agent
        next_states = {tuple(v): k for k, v in env.get_next_small_states().items()}
    else:
        # print("regular")
        agent = regular_agent
        next_states = {tuple(v): k for k, v in env.get_next_regular_states().items()}

    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done, lines_cleared = env.play(best_action[0], best_action[1], render=False)
    if lines_cleared == 4:
        tetris += 1
    elif lines_cleared == 3:
        almost_tetris += 1
    moves += 1

    tetrises_adaptive.append(tetris)
    almost_tetrises_adaptive.append(almost_tetris)
    scores_adaptive.append(env.score)

print("-----")
print(moves)
print(tetris)
print(env.score)


env = Tetris()
agent = regular_agent
done = False

tetris = 0
almost_tetris = 0
moves = 0
score = 0
tetrises_regular = []
almost_tetrises_regular = []
scores_regular = []
while not done and moves < 4000:
    next_states = {tuple(v): k for k, v in env.get_next_regular_states().items()}

    best_state = agent.best_state(next_states.keys())
    best_action = next_states[best_state]
    reward, done, lines_cleared = env.play(best_action[0], best_action[1], render=False)
    if lines_cleared == 4:
        tetris += 1
    elif lines_cleared == 3:
        almost_tetris += 1
    moves += 1

    tetrises_regular.append(tetris)
    almost_tetrises_regular.append(almost_tetris)
    scores_regular.append(env.score)
    

print("-----")
print(moves)
print(tetris)
print(env.score)



# Set style for all plots
# plt.style.use('seaborn')
xaxis = [i for i in range(1, 5001)]

# Plot 1: Tetrises
plt.figure(figsize=(12, 6))
plt.plot(tetrises_adaptive, label='Adaptive Strategy', color='#8884d8', linewidth=2)
plt.plot(tetrises_regular, label='Regular Strategy', color='#82ca9d', linewidth=2)
plt.title('Tetrises Achieved Over Time', fontsize=14, pad=20)
plt.xlabel('Moves')
plt.ylabel('Number of Tetrises')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Scores
plt.figure(figsize=(12, 6))
plt.plot(scores_adaptive, label='Adaptive Strategy', color='#8884d8', linewidth=2)
plt.plot(scores_regular, label='Regular Strategy', color='#82ca9d', linewidth=2)
plt.title('Score Progression', fontsize=14, pad=20)
plt.xlabel('Moves')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Combined Tetrises and Almost-Tetrises
combined_adaptive = [t + at for t, at in zip(tetrises_adaptive, almost_tetrises_adaptive)]
combined_regular = [t + at for t, at in zip(tetrises_regular, almost_tetrises_regular)]

plt.figure(figsize=(12, 6))
plt.plot(combined_adaptive, label='Adaptive Strategy', color='#8884d8', linewidth=2)
plt.plot(combined_regular, label='Regular Strategy', color='#82ca9d', linewidth=2)
plt.title('Combined Tetrises and Almost-Tetrises', fontsize=14, pad=20)
plt.xlabel('Moves')
plt.ylabel('Total Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Final Scores - Adaptive: {scores_adaptive[-1]:,}, Regular: {scores_regular[-1]:,}")
print(f"Total Tetrises - Adaptive: {tetrises_adaptive[-1]}, Regular: {tetrises_regular[-1]}")
print(f"Total Combined (Tetrises + Almost) - Adaptive: {combined_adaptive[-1]}, Regular: {combined_regular[-1]}")