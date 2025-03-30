from game_2048 import Game2048
from greedy_slover import GreedySolver


def main():
    game = Game2048()
    solver = GreedySolver(game)

    solver.solve()

if __name__ == "__main__":
    main()