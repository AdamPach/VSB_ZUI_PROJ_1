import time

from game.game_2048 import Game2048, GameState


class StatisticsRunner:
    def __init__(self, solver, number_runs=30):
        self.solver = solver
        self.number_runs = number_runs

    def run(self):
        print("Running games with solver for statistics")
        total_wins = 0
        total_losses = 0
        total_score = 0
        total_time = 0

        best_score = 0
        worse_score = 0

        for i in range(self.number_runs):
            print("====================================")
            print("Starting a new game")
            print(f"Game number: {i + 1} of {self.number_runs}")
            print("====================================")

            game = Game2048()

            start_time = time.perf_counter()

            final_state = self.solver(game, True).solve()

            end_time = time.perf_counter()

            total_wins += 1 if final_state == GameState.WIN else 0
            total_losses += 1 if final_state == GameState.LOSE else 0



            elapsed_time = end_time - start_time

            print(f"Game finished with state: {final_state}")
            print(f"Score: {game.score}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")

            total_score += game.score
            total_time += elapsed_time

            if game.score > best_score:
                best_score = game.score
            if game.score < worse_score or worse_score == 0:
                worse_score = game.score
            print("====================================")


        print("====================================")
        print("Statistics:")
        print(f"Total wins: {total_wins}")
        print(f"Total losses: {total_losses}")
        print(f"Total score: {total_score}")
        print(f"Average score: {total_score / self.number_runs:.2f}")
        print(f"Best score: {best_score}")
        print(f"Worst score: {worse_score}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time taken: {total_time / self.number_runs:.2f} seconds")
        print("====================================")
