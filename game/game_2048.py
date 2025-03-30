import random
import copy
from enum import Enum


class GameState(Enum):
    PLAYING = 0
    WIN = 1
    LOSE = 2

class Game2048:
    def __init__(self, size=4):
        """
        Inicializuje hru 2048 s danou velikostí hracího pole (default 4x4).
        """
        self.size = size
        # 0 bude reprezentovat prázdné pole
        self.board = [[0] * size for _ in range(size)]

        # Score
        self.score = 0

        # Umístíme na začátku dvě "dvojky"
        self._spawn_tile()
        self._spawn_tile()

    def clone(self):
        """
        Vytvoří a vrátí klon aktuální hry (včetně aktuálního skóre).
        """
        new_game = Game2048(self.size)
        # Zkopírujeme stav pole
        new_game.board = copy.deepcopy(self.board)
        # Zkopírujeme skóre
        new_game.score = self.score
        return new_game

    def get_board(self):
        """
        Vrátí kopii aktuálního stavu hracího pole jako 2D seznam.
        """
        return copy.deepcopy(self.board)

    def get_score(self):
        """
        Vrátí aktuální skóre.
        """
        return self.score

    def print_board(self):
        """
        Vytiskne hrací pole do konzole v textové podobě (včetně aktuálního skóre).
        """
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
        """
        Provede tah jedním ze směrů: "up", "down", "left", "right".
        Pokud se tah reálně projeví (pole se změní), dojde ke spawnutí nové dlaždice.
        """
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

        # Pokud se stav pole změnil, přidáme novou dlaždici (2 nebo 4)
        if self.board != old_board:
            moved = True
            self._spawn_tile()

        return moved

    def is_game_over(self):
        """
        Vyhodnotí stav hry (GameState):
         - WIN:   pokud na ploše existuje dlaždice 2048
         - PLAYING: pokud lze ještě táhnout (volné pole nebo možná fúze)
         - LOSE:  pokud nelze táhnout (deska je plná a nejdou spojit sousední dlaždice)
        """
        # 1) Kontrola, zda existuje dlaždice 2048 => WIN
        for row in self.board:
            if 2048 in row:
                return GameState.WIN

        # 2) Kontrola, jestli je možné táhnout => PLAYING
        #    a) pokud je kdekoliv nula
        for row in self.board:
            if 0 in row:
                return GameState.PLAYING

        #    b) pokud je kdekoliv možné sousední spojení
        #       (horizontální)
        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r][c] == self.board[r][c + 1]:
                    return GameState.PLAYING

        #       (vertikální)
        for c in range(self.size):
            for r in range(self.size - 1):
                if self.board[r][c] == self.board[r + 1][c]:
                    return GameState.PLAYING

        # 3) Pokud nic z výše uvedeného, pak LOSE
        return GameState.LOSE

    # ====================
    # Interní pomocné metody
    # ====================

    def _spawn_tile(self):
        """
        Přidá na náhodné volné místo v poli novou dlaždici (typicky 2 nebo 4).
        Ve standardní hře je dlaždice '2' pravděpodobnější,
        ale zde můžeme zjednodušit a dávat 2 a 4 v poměru 90:10.
        """
        empty_positions = [(r, c) for r in range(self.size)
                           for c in range(self.size)
                           if self.board[r][c] == 0]
        if not empty_positions:
            return

        r, c = random.choice(empty_positions)
        # Například: 10% šance na 4, jinak 2
        self.board[r][c] = 4 if random.random() < 0.1 else 2

    def _compress_and_merge_line(self, line):
        """
        Zpracuje (posune) jeden řádek (list) směrem doleva:
         1) Vyhodí nuly a seřadí čísla doleva
         2) Sloučí sousední stejná čísla (přičte je k self.score)
         3) Výsledek doplní nulami vpravo
        Příklad: line = [2, 2, 4, 0] -> [4, 4, 0, 0], score += 4
        """
        # Odfiltrujeme nuly
        filtered = [x for x in line if x != 0]

        merged_line = []
        skip = False
        for i in range(len(filtered)):
            if skip:
                skip = False
                continue

            if i < len(filtered) - 1 and filtered[i] == filtered[i + 1]:
                # Spojíme dvojice
                new_value = filtered[i] * 2
                merged_line.append(new_value)

                # Přičteme nově vzniklou hodnotu do skóre
                self.score += new_value

                skip = True
            else:
                merged_line.append(filtered[i])

        # Doplníme zbytek nulami
        merged_line.extend([0] * (len(line) - len(merged_line)))
        return merged_line

    def _move_left(self):
        """
        Sloučí a posune hrací desku doleva po řádcích.
        """
        new_board = []
        for row in self.board:
            merged = self._compress_and_merge_line(row)
            new_board.append(merged)
        self.board = new_board

    def _move_right(self):
        """
        Sloučí a posune hrací desku doprava po řádcích.
        (Vyřešíme jednoduše reversem, move_left, reversem zpět.)
        """
        new_board = []
        for row in self.board:
            reversed_row = row[::-1]
            merged = self._compress_and_merge_line(reversed_row)
            new_board.append(merged[::-1])
        self.board = new_board

    def _move_up(self):
        """
        Sloučí a posune hrací desku nahoru po sloupcích.
        (Implementačně transponujeme, posuneme doleva, transponujeme zpět.)
        """
        self._transpose()
        self._move_left()
        self._transpose()

    def _move_down(self):
        """
        Sloučí a posune hrací desku dolů po sloupcích.
        (Implementačně transponujeme, posuneme doprava, transponujeme zpět.)
        """
        self._transpose()
        self._move_right()
        self._transpose()

    def _transpose(self):
        """
        Transponuje matici (převrátí řádky a sloupce).
        """
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
