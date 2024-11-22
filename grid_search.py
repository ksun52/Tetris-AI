from MCTS_grant import TetrisAI

import numpy as np
from tqdm import tqdm  # For progress bar

# Number of iterations for the random search
num_iterations = 1000  # You can adjust this number

# Variables to store the best score and corresponding weights
best_score = float('-inf')
best_weights = None

# Define the range for each of the 5 heuristic weights
weight_ranges = [(-1, 1)] * 5  # Adjust ranges if needed

# Loop over the number of iterations with a progress bar
for i in tqdm(range(num_iterations), desc="Optimizing", unit="iteration"):
    # Generate a random set of heuristic weights within the specified ranges
    heuristic_weights = np.random.uniform(-1, 1, size=5)
    
    # Create a TetrisAI instance with the random weights
    ai = TetrisAI(render=False, heuristic_weights=heuristic_weights)
    
    # Play the game and obtain the score
    score = ai.play_game()
    
    # If the score is better than the current best, update the best score and weights
    if score > best_score:
        best_score = score
        best_weights = heuristic_weights
        # Print the new best score and weights without disrupting the progress bar
        tqdm.write(f"Iteration {i}: New best score: {best_score} with weights: {best_weights}")

# After all iterations, print the best score and corresponding heuristic weights
print("\nOptimization Complete")
print(f"Best Score: {best_score}")
print(f"Best Heuristic Weights: {best_weights}")
