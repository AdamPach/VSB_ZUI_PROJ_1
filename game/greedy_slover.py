from game_2048 import GameState


class GreedySolver:
    def __init__(self, game):
        self.game = game

    def solve(self):
        state = self.game.is_game_over()

        while state == GameState.PLAYING:
            best_move = None
            best_score = -1

            for move in ["left", "right", "up", "down"]:
                new_game = self.game.clone()
                valid_move = new_game.move(move)

                if valid_move and new_game.get_score() > best_score:
                    best_move = move
                    best_score = new_game.get_score()

            self.game.move(best_move)
            print(f"Moving {best_move}")
            self.game.print_board()
            state = self.game.is_game_over()

        return state