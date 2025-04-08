import random
import time

from game.game_2048 import GameState

class MonteCarloSolver:
    def __init__(self, game, display=False, simulations_per_move=200, max_depth=200, time_limit=20):
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
        self.display = display

        # Possible moves
        self.directions = ["up", "down", "left", "right"]

        # Weight parameters for the heuristic evaluation
        self.weights = {
            "score": 1.0,  # Game score
            "empty_tiles": 2.5,  # Number of empty tiles (more is better)
            "max_tile": 1.0,  # Value of the highest tile
            "smoothness": 0.1,  # Measure of how smooth the board is (adjacent tiles similar)
            "monotonicity": 1.0,  # Measure of monotonic arrangement
            "corner_max": 2.0  # Bonus for having max tile in a corner
        }

    def solve(self):
        state = self.game.is_game_over()

        while state == GameState.PLAYING:
            best_move = self.get_best_move()
            valid_move = self.game.move(best_move)

            if not valid_move:
                break

            state = self.game.is_game_over()

            if self.display:
                print(f"Moving {best_move}")
                self.game.print_board()

        return state

    def get_best_move(self):
        """
        Determine the best move from the current game state.

        Returns:
            The best direction to move ('up', 'down', 'left', 'right')
        """
        start_time = time.time()
        best_move = None
        best_score = float('-inf')

        # Try each possible move
        for direction in self.directions:
            game_clone = self.game.clone()

            # Check if the move is valid
            if not game_clone.move(direction):
                continue

            # If valid, run simulations from this state
            move_score = self.run_simulations(game_clone)

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
        Run multiple random simulations from the given game state.

        Args:
            game: The Game2048 instance to simulate from

        Returns:
            The average evaluation score from all simulations
        """
        total_score = 0

        for _ in range(self.simulations_per_move):
            # Clone the game to avoid modifying the original
            sim_game = game.clone()

            # Run a random simulation
            self.random_playout(sim_game)

            # Evaluate the final state
            evaluation = self.evaluate_position(sim_game)
            total_score += evaluation

        # Return average score
        return total_score / self.simulations_per_move

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

        # 2. Empty tiles
        empty_count = sum(row.count(0) for row in board)
        empty_value = empty_count * self.weights["empty_tiles"]

        # 3. Max tile
        max_tile = max(max(row) for row in board)
        max_tile_value = max_tile * self.weights["max_tile"]

        # 4. Smoothness (penalize differences between adjacent tiles)
        smoothness = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    continue

                # Check right neighbor
                if c + 1 < size and board[r][c + 1] != 0:
                    smoothness -= abs(board[r][c] - board[r][c + 1])

                # Check down neighbor
                if r + 1 < size and board[r + 1][c] != 0:
                    smoothness -= abs(board[r][c] - board[r + 1][c])

        smoothness_value = smoothness * self.weights["smoothness"]

        # 5. Monotonicity (prefer tiles to be in increasing/decreasing order)
        monotonicity = 0

        # Check horizontal monotonicity
        for r in range(size):
            # Left to right
            mono_left = 0
            for c in range(1, size):
                if board[r][c - 1] != 0 and board[r][c] != 0:
                    if board[r][c - 1] >= board[r][c]:
                        mono_left += 1

            # Right to left
            mono_right = 0
            for c in range(size - 1, 0, -1):
                if board[r][c - 1] != 0 and board[r][c] != 0:
                    if board[r][c - 1] <= board[r][c]:
                        mono_right += 1

            monotonicity += max(mono_left, mono_right)

        # Check vertical monotonicity
        for c in range(size):
            # Top to bottom
            mono_top = 0
            for r in range(1, size):
                if board[r - 1][c] != 0 and board[r][c] != 0:
                    if board[r - 1][c] >= board[r][c]:
                        mono_top += 1

            # Bottom to top
            mono_bottom = 0
            for r in range(size - 1, 0, -1):
                if board[r - 1][c] != 0 and board[r][c] != 0:
                    if board[r - 1][c] <= board[r][c]:
                        mono_bottom += 1

            monotonicity += max(mono_top, mono_bottom)

        monotonicity_value = monotonicity * self.weights["monotonicity"]

        # 6. Bonus for max tile in corner
        corner_max_value = 0
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]

        for r, c in corners:
            if board[r][c] == max_tile:
                corner_max_value = max_tile * self.weights["corner_max"]
                break

        # Combine all heuristics
        total_evaluation = (
                score_value +
                empty_value +
                max_tile_value +
                smoothness_value +
                monotonicity_value +
                corner_max_value
        )

        return total_evaluation

class MonteCarloSimpleSolver:
    """
    A Monte Carlo solver for the 2048 game.

    This solver uses random simulations to determine the best move without
    using any heuristics. For each possible move, it simulates multiple random
    games to completion and chooses the move that leads to the highest average score.
    """

    def __init__(self, game, display = False):
        """
        Initialize the solver with an optional game instance.

        Args:
            game: An instance of Game2048 (optional)
        """
        self.game = game
        self.directions = ["up", "down", "left", "right"]
        self.display = display

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

            if self.display:
                print(f"Moving {best_move}")
                self.game.print_board()

        return state

    def find_best_move(self, num_simulations=200):
        """
        Find the best move for the given game state using Monte Carlo simulation.

        Args:
            game: Game state to analyze (defaults to the instance's game)
            num_simulations: Number of random simulations to run per possible move

        Returns:
            The best move direction ("up", "down", "left", or "right")
        """
        best_direction = None
        best_score = float('-inf')
        results = {}

        # Evaluate each possible move
        for direction in self.directions:
            # Create a clone of the current game
            game_clone = self.game.clone()

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