from game_2048 import Game2048
from greedy_slover import GreedySolver
from game_2048 import GameState


def main():
    game = Game2048()
    solver = GreedySolver(game)

    final_state = solver.solve()

    if final_state == GameState.WIN:
        print("You won!")
    else:
        print("You lost!")

    game.print_statistics()

if __name__ == "__main__":
    main()