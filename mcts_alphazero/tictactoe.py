"""Small immutable Tic-Tac-Toe environment shared by all search code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PLAYER_X = 1
PLAYER_O = -1
EMPTY = 0

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True)
class TicTacToeState:
    """Immutable board state with `1` for X, `-1` for O, and `0` for empty."""

    board: tuple[int, ...] = (EMPTY,) * 9
    player: int = PLAYER_X

    def __post_init__(self) -> None:
        if len(self.board) != 9:
            raise ValueError("Tic-Tac-Toe board must have exactly 9 cells")
        if self.player not in (PLAYER_X, PLAYER_O):
            raise ValueError("player must be 1 (X) or -1 (O)")
        if any(cell not in (PLAYER_X, PLAYER_O, EMPTY) for cell in self.board):
            raise ValueError("board cells must be 1, -1, or 0")

    def legal_actions(self) -> list[int]:
        """Return all currently empty cell indices."""

        if self.is_terminal():
            return []
        return [idx for idx, cell in enumerate(self.board) if cell == EMPTY]

    def apply(self, action: int) -> "TicTacToeState":
        """Return the next state after placing the current player's mark."""

        if action not in range(9):
            raise ValueError("action must be an integer cell index from 0 to 8")
        if self.board[action] != EMPTY:
            raise ValueError(f"cell {action} is already occupied")
        if self.is_terminal():
            raise ValueError("cannot apply an action to a terminal state")

        board = list(self.board)
        board[action] = self.player
        return TicTacToeState(tuple(board), -self.player)

    def winner(self) -> int | None:
        """Return `1`, `-1`, `0` for draw, or `None` if the game is unfinished."""

        for a, b, c in WIN_LINES:
            line_sum = self.board[a] + self.board[b] + self.board[c]
            if line_sum == 3:
                return PLAYER_X
            if line_sum == -3:
                return PLAYER_O
        if all(cell != EMPTY for cell in self.board):
            return 0
        return None

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def result_for_player(self, player: int) -> float:
        """Return terminal result from `player` perspective: win=1, draw=0, loss=-1."""

        winner = self.winner()
        if winner is None:
            raise ValueError("result is only defined for terminal states")
        if winner == 0:
            return 0.0
        return 1.0 if winner == player else -1.0

    def encode_for_current_player(self) -> list[float]:
        """Encode the board from the player-to-move perspective."""

        return [float(cell * self.player) for cell in self.board]

    @classmethod
    def from_rows(cls, rows: Iterable[str], player: int) -> "TicTacToeState":
        """Build a state from rows using X, O, and . characters."""

        mapping = {"X": PLAYER_X, "O": PLAYER_O, ".": EMPTY}
        cells = [mapping[ch] for row in rows for ch in row]
        return cls(tuple(cells), player)

    def render(self) -> str:
        symbols = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        rows = ["".join(symbols[self.board[i + j]] for j in range(3)) for i in (0, 3, 6)]
        return "\n".join(rows)
