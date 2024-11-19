### Adjust C value
Make the C value for UCB adaptive with respect to the current depth. Since game scores get higher as depth increases, we should weight exploration more to make up for increases utility scores.adaptive

### Heuristics to Motivate Multiline Break Moves
If a state is configured such that it is in position for a multi-line break, we should value that in the state in a more direct way. We notice right now that the Deep Q model doesn't seem to value these moves enough. This could be done by having a heuristic that adds value exponentially with respect to clearable lines (e.g. if there's a state with a Tetris available and a 3-line break then the value of this heuristic would be 4^2 + 3^2 = 25)