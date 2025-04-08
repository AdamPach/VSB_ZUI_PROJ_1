from game.stats_runner import StatisticsRunner
from monte_carlo_solver import MonteCarloSolver, MonteCarloSimpleSolver
from game_2048 import Game2048
from greedy_slover import GreedySolver
from game_2048 import GameState


def main():
    runner = StatisticsRunner(MonteCarloSimpleSolver, 30)
    runner.run()

if __name__ == "__main__":
    main()