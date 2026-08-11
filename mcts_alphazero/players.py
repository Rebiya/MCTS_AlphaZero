"""Simple player wrappers for experiments."""

from __future__ import annotations

import random
from typing import Protocol

from .mcts import PureMCTS
from .tictactoe import TicTacToeState


class Player(Protocol):
    def choose_action(self, state: TicTacToeState) -> int:
        ...

    def update_with_action(self, action: int) -> None:
        ...


class RandomPlayer:
    """Uniformly random legal-action player."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def choose_action(self, state: TicTacToeState) -> int:
        return self.random.choice(state.legal_actions())

    def update_with_action(self, action: int) -> None:
        return None


class MCTSPlayer:
    """Player adapter around `PureMCTS`."""

    def __init__(self, mcts: PureMCTS) -> None:
        self.mcts = mcts

    def choose_action(self, state: TicTacToeState) -> int:
        return self.mcts.choose_action(state)

    def update_with_action(self, action: int) -> None:
        self.mcts.update_with_action(action)


def play_game(x_player: Player, o_player: Player) -> int:
    """Play one game and return winner: 1 for X, -1 for O, 0 for draw."""

    state = TicTacToeState()
    players = {1: x_player, -1: o_player}
    while not state.is_terminal():
        action = players[state.player].choose_action(state)
        if action not in state.legal_actions():
            raise ValueError(f"illegal action selected: {action}")
        state = state.apply(action)
        x_player.update_with_action(action)
        o_player.update_with_action(action)
    winner = state.winner()
    assert winner is not None
    return winner
