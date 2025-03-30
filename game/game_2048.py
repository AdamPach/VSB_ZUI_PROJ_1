import random
import copy
from enum import Enum


class GameState(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.board = [[0] * size for _ in range(size)]

        self.score = 0

        self._spawn_tile()
        self._spawn_tile()

    def clone(self):
        new_game = Game2048(self.size)
        new_game.board = copy.deepcopy(self.board)
        new_game.score = self.score
        return new_game

    def get_board(self):
        return copy.deepcopy(self.board)

    def get_score(self):
        return self.score

    def print_board(self):
        print(f"Score: {self.score}")
        size = self.size

        for r in range(size):
            row_str = []
            for c in range(size):
                value = self.board[r][c]
                if value == 0:
                    row_str.append("empty")
                else:
                    row_str.append(str(value))
            print(" | ".join(f"{x:>5}" for x in row_str))

            if r < size - 1:
                print("-----+" * (size - 1) + "-----")

    def move(self, direction):
        old_board = self.get_board()

        if direction == "left":
            self._move_left()
        elif direction == "right":
            self._move_right()
        elif direction == "up":
            self._move_up()
        elif direction == "down":
            self._move_down()
        else:
            raise ValueError("Neznámý směr pohybu! Použijte: up, down, left, right.")

        moved = False

        if self.board != old_board:
            moved = True
            self._spawn_tile()

        return moved

    def is_game_over(self):
        for row in self.board:
            if 2048 in row:
                return GameState.WIN

        for row in self.board:
            if 0 in row:
                return GameState.PLAYING

        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r][c] == self.board[r][c + 1]:
                    return GameState.PLAYING

        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r][c] == self.board[r + 1][c]:
                    return GameState.PLAYING

        return GameState.LOSE

    def print_statistics(self):
        print(f"Score: {self.score}")

    def _spawn_tile(self):
        empty_positions = [(r, c) for r in range(self.size)
                           for c in range(self.size)
                           if self.board[r][c] == 0]
        if not empty_positions:
            return

        r, c = random.choice(empty_positions)
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    def _compress_and_merge_line(self, line):
        filtered = [x for x in line if x != 0]

        merged_line = []
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue

            if i < len(filtered) - 1 and filtered[i] == filtered[i + 1]:
                new_value = filtered[i] * 2
                merged_line.append(new_value)

                self.score += new_value

                skip = True
            else:
                merged_line.append(filtered[i])

        merged_line.extend([0] * (len(line) - len(merged_line)))
        return merged_line

    def _move_left(self):
        new_board = []
        for row in self.board:
            merged = self._compress_and_merge_line(row)
            new_board.append(merged)
        self.board = new_board

    def _move_right(self):
        new_board = []
        for row in self.board:
            reversed_row = row[::-1]
            merged = self._compress_and_merge_line(reversed_row)
            new_board.append(merged[::-1])
        self.board = new_board

    def _move_up(self):
        self._transpose()
        self._move_left()
        self._transpose()

    def _move_down(self):
        self._transpose()
        self._move_right()
        self._transpose()

    def _transpose(self):
        self.board = [list(row) for row in zip(*self.board)]


if __name__ == "__main__":
    # Krátký demonstrační příklad, jak hra funguje.
    game = Game2048(size=4)
    game.print_board()
    print("-----------")

    # Ukázka pohybů a zjišťování stavu hry
    for direction in ["left", "up", "right", "down"]:
        print(f"Táhneme: {direction}")
        game.move(direction)
        game.print_board()
        state = game.is_game_over()  # Nyní vrací GameState
        print("Stav hry:", state.name)  # nebo state.value
        print("-----------")

    # Vytvoření klonu hry
    cloned_game = game.clone()
    print("Klon hry (se stejným skóre):")
    cloned_game.print_board()
    print("Stav klonu hry:", cloned_game.is_game_over().name)
