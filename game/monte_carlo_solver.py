import random
import time

from game_2048 import Game2048, GameState

class MonteCarloSolver:
    def __init__(self, game, simulations_per_move=300, max_depth=300, time_limit=20):
        """
        Initialize the Monte Carlo solver.

        Args:
            game: The Game2048 instance to solve
            simulations_per_move: Number of random simulations per candidate move
            max_depth: Maximum number of moves in a simulation
            time_limit: Maximum time (in seconds) to spend deciding a move
        """
        self.game = game
        self.simulations_per_move = simulations_per_move
        self.max_depth = max_depth
        self.time_limit = time_limit

        # Possible moves
        self.directions = ["up", "down", "left", "right"]

        # Weight parameters for the heuristic evaluation
        self.weights = {
            "score": 0.5,  # Game score (reduced importance)
            "empty_tiles": 5.0,  # Number of empty tiles (critical importance)
            "max_tile": 1.0,  # Value of the highest tile
            "smoothness": 0.4,  # Measure of how smooth the board is (increased importance)
            "monotonicity": 2.5,  # Measure of monotonic arrangement (significantly increased)
            "corner_max": 1.5,  # Bonus for having max tile in a corner (slightly reduced)
            "gradient": 2.0,  # New: Reward decreasing values from max tile corner
            "merge_potential": 1.5,  # New: Reward potential merges
            "snake_pattern": 3.0  # New: Reward snake-like arrangement
        }

        # Define snake pattern templates (top-right focused)
        self.snake_templates = [
            # Snake pattern from top-right corner (most common 2048 strategy)
            [
                [3, 2, 1, 0],
                [4, 5, 6, 7],
                [11, 10, 9, 8],
                [12, 13, 14, 15]
            ],
            # Alternative snake from top-left
            [
                [0, 1, 2, 3],
                [7, 6, 5, 4],
                [8, 9, 10, 11],
                [15, 14, 13, 12]
            ]
        ]

    def solve(self):
        """
        Solve the game using Monte Carlo Tree Search.

        Returns:
            The final game state (win or lose)
        """
        state = self.game.is_game_over()

        while state == GameState.PLAYING:
            best_move = self.get_best_move()
            valid_move = self.game.move(best_move)

            if not valid_move:
                break

            self.game.print_board()
            state = self.game.is_game_over()

        return state

    def get_best_move(self):
        """
        Determine the best move from the current game state with advanced lookahead.

        Returns:
            The best direction to move ('up', 'down', 'left', 'right')
        """
        start_time = time.time()
        best_move = None
        best_score = float('-inf')

        # Boosting factors to prefer certain moves in critical situations
        move_preferences = {
            "up": 1.02,  # Slightly prefer up moves to maintain top corner strategy
            "left": 1.01,  # Then prefer left
            "right": 1.00,  # Neutral preference
            "down": 0.97  # Avoid down moves which can disrupt corner placement
        }

        # Current board state for analysis
        current_board = self.game.get_board()
        size = self.game.size
        max_tile = max(max(row) for row in current_board)
        empty_count = sum(row.count(0) for row in current_board)

        # Check if we're in a critical state (few empty tiles or high value tile)
        critical_state = empty_count <= 2 or max_tile >= 512

        # Try each possible move
        move_scores = {}

        for direction in self.directions:
            game_clone = self.game.clone()

            # Check if the move is valid
            if not game_clone.move(direction):
                continue

            # Direct evaluation of resulting position
            direct_eval = self.evaluate_position(game_clone)

            # For critical states, use deeper lookahead
            if critical_state:
                # Look one move deeper to better evaluate this position
                next_move_scores = []

                # Try each possible follow-up move
                for next_direction in self.directions:
                    next_game = game_clone.clone()

                    if next_game.move(next_direction):
                        # Evaluate after two moves
                        next_eval = self.evaluate_position(next_game)
                        next_move_scores.append(next_eval)

                # If we found valid follow-up moves, consider their max score
                if next_move_scores:
                    # Consider both the immediate position and best follow-up
                    direct_eval = direct_eval * 0.7 + max(next_move_scores) * 0.3

            # Run simulations from this state
            sim_score = self.run_simulations(game_clone)

            # Combine direct evaluation and simulation results
            # In critical states, give more weight to direct evaluation
            if critical_state:
                move_score = direct_eval * 0.6 + sim_score * 0.4
            else:
                move_score = direct_eval * 0.2 + sim_score * 0.8

            # Apply move preference factor
            move_score *= move_preferences[direction]

            # Track all scores
            move_scores[direction] = move_score

            if move_score > best_score:
                best_score = move_score
                best_move = direction

            # Check if we're out of time
            if time.time() - start_time > self.time_limit:
                break

        # If no move was found (shouldn't happen with a valid game), pick randomly
        if best_move is None:
            valid_moves = []
            for direction in self.directions:
                game_clone = self.game.clone()
                if game_clone.move(direction):
                    valid_moves.append(direction)

            if valid_moves:
                best_move = random.choice(valid_moves)
            else:
                best_move = random.choice(self.directions)

        return best_move

    def run_simulations(self, game):
        """
        Run multiple intelligent simulations from the given game state.

        Args:
            game: The Game2048 instance to simulate from

        Returns:
            The average evaluation score from all simulations
        """
        total_score = 0
        max_score = float('-inf')
        scores = []

        # Get initial board state
        initial_board = game.get_board()
        max_tile = max(max(row) for row in initial_board)
        empty_count = sum(row.count(0) for row in initial_board)

        # Adaptive simulation depth - deeper for critical positions
        if max_tile >= 512:
            # For high tiles, do fewer but deeper simulations
            sim_count = max(10, self.simulations_per_move // 2)
            depth_factor = 2.0
        elif empty_count <= 2:
            # For crowded boards, prioritize immediate survival
            sim_count = max(20, int(self.simulations_per_move // 1.5))
            depth_factor = 1.5
        else:
            # Normal case
            sim_count = self.simulations_per_move
            depth_factor = 1.0

        for _ in range(sim_count):
            # Clone the game to avoid modifying the original
            sim_game = game.clone()

            # Run a guided simulation instead of purely random
            if random.random() < 0.7:  # 70% chance of guided simulation
                self.guided_playout(sim_game, depth_factor)
            else:
                self.random_playout(sim_game)

            # Evaluate the final state
            evaluation = self.evaluate_position(sim_game)
            total_score += evaluation
            scores.append(evaluation)

            # Track best simulation
            max_score = max(max_score, evaluation)

        # Calculate statistics from simulations
        avg_score = total_score / sim_count

        # Return a weighted combination of average and max score
        # This helps avoid good moves being missed due to averaging with bad random outcomes
        return 0.6 * avg_score + 0.4 * max_score

    def guided_playout(self, game, depth_factor=1.0):
        """
        Play a guided game using simplified heuristics.

        Args:
            game: The Game2048 instance to play
            depth_factor: Multiplier for max_depth
        """
        max_depth = int(self.max_depth * depth_factor)
        moves = 0

        # Simple move heuristic priorities
        move_priorities = ["up", "left", "right", "down"]

        while moves < max_depth:
            state = game.is_game_over()
            if state != GameState.PLAYING:
                break

            # Try moves in priority order first
            valid_move = False

            # Get current board
            board = game.get_board()
            size = game.size
            max_tile = max(max(row) for row in board)

            # Find the corner with max value
            corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            max_corner = None

            for r, c in corners:
                if board[r][c] == max_tile:
                    max_corner = (r, c)
                    break

            # Adjust priorities based on max tile position
            if max_corner:
                r, c = max_corner
                if r == 0 and c == 0:  # Top-left
                    move_priorities = ["left", "up", "right", "down"]
                elif r == 0 and c == size - 1:  # Top-right
                    move_priorities = ["up", "right", "left", "down"]
                elif r == size - 1 and c == 0:  # Bottom-left
                    move_priorities = ["left", "down", "right", "up"]
                else:  # Bottom-right
                    move_priorities = ["right", "down", "left", "up"]

            # Try moves in priority order
            for direction in move_priorities:
                if game.move(direction):
                    valid_move = True
                    break

            # If no priority move worked, try random moves
            if not valid_move:
                shuffled_directions = random.sample(self.directions, len(self.directions))
                for direction in shuffled_directions:
                    if game.move(direction):
                        valid_move = True
                        break

            if not valid_move:
                break

            moves += 1

    def random_playout(self, game):
        """
        Play a random game from the current state until game over or max depth.

        Args:
            game: The Game2048 instance to play
        """
        moves = 0

        while moves < self.max_depth:
            # Game over check
            state = game.is_game_over()
            if state != GameState.PLAYING:
                break

            # Try random directions until a valid move is found
            valid_move = False
            shuffled_directions = random.sample(self.directions, len(self.directions))

            for direction in shuffled_directions:
                if game.move(direction):
                    valid_move = True
                    break

            if not valid_move:
                # No valid moves found (shouldn't happen, but just in case)
                break

            moves += 1

    def evaluate_position(self, game):
        """
        Evaluate a game position using multiple heuristics.

        Args:
            game: The Game2048 instance to evaluate

        Returns:
            A score indicating how good the position is
        """
        board = game.get_board()
        size = game.size

        # 1. Game score
        score_value = game.get_score() * self.weights["score"]

        # 2. Empty tiles (critical)
        empty_count = sum(row.count(0) for row in board)
        # Exponential weighting - empty tiles become increasingly valuable
        empty_value = (2 ** empty_count) * self.weights["empty_tiles"]

        # 3. Max tile
        max_tile = max(max(row) for row in board)
        max_tile_value = max_tile * self.weights["max_tile"]

        # Find position of max tile
        max_pos = None
        for r in range(size):
            for c in range(size):
                if board[r][c] == max_tile:
                    max_pos = (r, c)
                    break
            if max_pos:
                break

        # 4. Improved smoothness (penalize large differences between adjacent tiles)
        smoothness = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue

                cell_value = board[r][c]

                # Check right neighbor
                if c + 1 < size and board[r][c + 1] != 0:
                    # Log-based difference penalty for smoother gradient
                    if cell_value != 0 and board[r][c + 1] != 0:
                        diff = abs(cell_value - board[r][c + 1])
                        # Bigger penalty for large value differences
                        if diff > 0:
                            log_diff = 1 + abs(cell_value - board[r][c + 1]) / max(cell_value, board[r][c + 1])
                            smoothness -= log_diff * 2

                # Check down neighbor
                if r + 1 < size and board[r + 1][c] != 0:
                    if cell_value != 0 and board[r + 1][c] != 0:
                        diff = abs(cell_value - board[r + 1][c])
                        if diff > 0:
                            log_diff = 1 + abs(cell_value - board[r + 1][c]) / max(cell_value, board[r + 1][c])
                            smoothness -= log_diff * 2

        smoothness_value = smoothness * self.weights["smoothness"]

        # 5. Enhanced monotonicity (prefer decreasing values from the corner with max tile)
        monotonicity = 0

        # Horizontal monotonicity
        for r in range(size):
            # Left to right and right to left monotonicity
            mono_left = 0
            mono_right = 0

            for c in range(1, size):
                # Left to right - increasing
                if board[r][c - 1] != 0 and board[r][c] != 0:
                    if board[r][c - 1] <= board[r][c]:
                        mono_left += 1
                    else:
                        # Penalize drops in value more heavily
                        mono_left -= 1

                # Right to left - increasing
                if board[r][size - c] != 0 and board[r][size - c - 1] != 0:
                    if board[r][size - c] <= board[r][size - c - 1]:
                        mono_right += 1
                    else:
                        mono_right -= 1

            monotonicity += max(mono_left, mono_right)

        # Vertical monotonicity
        for c in range(size):
            # Top to bottom and bottom to top
            mono_top = 0
            mono_bottom = 0

            for r in range(1, size):
                # Top to bottom - increasing
                if board[r - 1][c] != 0 and board[r][c] != 0:
                    if board[r - 1][c] <= board[r][c]:
                        mono_top += 1
                    else:
                        mono_top -= 1

                # Bottom to top - increasing
                if board[size - r][c] != 0 and board[size - r - 1][c] != 0:
                    if board[size - r][c] <= board[size - r - 1][c]:
                        mono_bottom += 1
                    else:
                        mono_bottom -= 1

            monotonicity += max(mono_top, mono_bottom)

        monotonicity_value = monotonicity * self.weights["monotonicity"]

        # 6. Enhanced corner max tile bonus
        corner_max_value = 0
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]

        # Find which corner the max tile is in (if any)
        max_corner = None
        for r, c in corners:
            if board[r][c] == max_tile:
                max_corner = (r, c)
                # Bonus scales with the value of the tile
                corner_max_value = max_tile * self.weights["corner_max"]
                break

        # 7. NEW: Gradient from max tile in corner
        gradient_value = 0
        if max_corner:
            # Reward decreasing values from the corner with max tile
            r_corner, c_corner = max_corner
            gradient_sum = 0

            # Choose direction based on which corner has max
            r_dir = 1 if r_corner == 0 else -1
            c_dir = 1 if c_corner == 0 else -1

            # Previous value (starting with the max)
            prev_val = max_tile

            # Check each position in outward spiral from corner
            for dist in range(1, size * 2):
                # Move outward from corner in spiral
                for dr in range(min(dist, size)):
                    for dc in range(min(dist - dr, size)):
                        r = r_corner + dr * r_dir
                        c = c_corner + dc * c_dir

                        # Check if in bounds
                        if 0 <= r < size and 0 <= c < size:
                            if board[r][c] == 0:
                                continue

                            # Reward decreasing values
                            if board[r][c] <= prev_val:
                                gradient_sum += 1
                                prev_val = board[r][c]
                            else:
                                # Penalize increasing values
                                gradient_sum -= 1

            gradient_value = gradient_sum * self.weights["gradient"]

        # 8. NEW: Merge potential (adjacent same-value tiles)
        merge_potential = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue

                # Check horizontal merge possibility
                if c + 1 < size and board[r][c] == board[r][c + 1] and board[r][c] != 0:
                    # Higher value tiles get exponentially more merge value
                    merge_potential += board[r][c]

                # Check vertical merge possibility
                if r + 1 < size and board[r][c] == board[r + 1][c] and board[r][c] != 0:
                    merge_potential += board[r][c]

        merge_value = merge_potential * self.weights["merge_potential"]

        # 9. NEW: Snake pattern evaluation
        snake_value = 0

        # Flatten the board for easier snake pattern matching
        flat_board = [board[r][c] for r in range(size) for c in range(size)]

        # Sort the values for comparisons
        sorted_values = sorted([v for v in flat_board if v != 0], reverse=True)

        # Try each snake template
        for template in self.snake_templates:
            flat_template = [template[r][c] for r in range(size) for c in range(size)]

            # Count tiles that follow the snake pattern
            snake_match = 0
            for idx, val_rank in enumerate(flat_template):
                if idx >= len(flat_board) or flat_board[idx] == 0:
                    continue

                # Check if value follows the expected rank in the pattern
                expected_rank = val_rank
                if expected_rank < len(sorted_values):
                    # Reward if this position has appropriate value according to pattern
                    actual_rank = sorted_values.index(flat_board[idx]) if flat_board[idx] in sorted_values else -1

                    if abs(actual_rank - expected_rank) <= 2:  # Allow some tolerance
                        snake_match += 1

                        # Bonus for highest values in the right positions
                        if expected_rank == 0 and actual_rank == 0:
                            snake_match += 3
                        elif expected_rank <= 3 and actual_rank <= 3:
                            snake_match += 1

            snake_value = max(snake_value, snake_match * self.weights["snake_pattern"])

        # Combine all heuristics
        total_evaluation = (
                score_value +
                empty_value +
                max_tile_value +
                smoothness_value +
                monotonicity_value +
                corner_max_value +
                gradient_value +
                merge_value +
                snake_value
        )

        return total_evaluation


class MonteCarloSimpleSolver:
    """
    A Monte Carlo solver for the 2048 game.

    This solver uses random simulations to determine the best move without
    using any heuristics. For each possible move, it simulates multiple random
    games to completion and chooses the move that leads to the highest average score.
    """

    def __init__(self, game=None):
        """
        Initialize the solver with an optional game instance.

        Args:
            game: An instance of Game2048 (optional)
        """
        self.game = game if game is not None else Game2048()
        self.directions = ["up", "down", "left", "right"]

    def solve(self):
        """
        Solve the game using Monte Carlo simulation.

        Returns:
            The final game state (win or lose)
        """
        state = self.game.is_game_over()

        while state == GameState.PLAYING:
            best_move = self.find_best_move()
            valid_move = self.game.move(best_move)

            if not valid_move:
                break

            state = self.game.is_game_over()
            self.game.print_board()

        return state

    def find_best_move(self, game=None, num_simulations=200):
        """
        Find the best move for the given game state using Monte Carlo simulation.

        Args:
            game: Game state to analyze (defaults to the instance's game)
            num_simulations: Number of random simulations to run per possible move

        Returns:
            The best move direction ("up", "down", "left", or "right")
        """
        game = game if game is not None else self.game
        best_direction = None
        best_score = float('-inf')
        results = {}

        # Evaluate each possible move
        for direction in self.directions:
            # Create a clone of the current game
            game_clone = game.clone()

            # Try the move
            moved = game_clone.move(direction)

            # If move is not valid, skip it
            if not moved:
                continue

            # Run simulations from this new state
            total_score = 0

            for _ in range(num_simulations):
                # Clone again for each simulation
                sim_clone = game_clone.clone()
                # Run a random simulation to completion
                final_score = self._simulate_random_game(sim_clone)
                total_score += final_score

            # Calculate average score for this move
            average_score = total_score / num_simulations
            results[direction] = average_score

            # Update best move if this one is better
            if average_score > best_score:
                best_score = average_score
                best_direction = direction

        # If no valid moves found (shouldn't happen in normal gameplay)
        if best_direction is None:
            return random.choice(self.directions)

        return best_direction

    def _simulate_random_game(self, game):
        """
        Simulate a game to completion using random moves.

        Args:
            game: Game state to start from

        Returns:
            Final score of the simulated game
        """
        max_moves = 1000  # Safety limit to prevent infinite loops
        moves_made = 0

        # Continue until game is over or max moves reached
        while game.is_game_over() == GameState.PLAYING and moves_made < max_moves:
            # Try all moves in a random order until one works
            random_directions = random.sample(self.directions, len(self.directions))
            made_move = False

            for direction in random_directions:
                if game.move(direction):
                    made_move = True
                    moves_made += 1
                    break

            # If no valid moves, the game should be over
            if not made_move:
                break

        # Return the final score
        return game.get_score()