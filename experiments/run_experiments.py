"""Run small MCTS vs AlphaZero-style Tic-Tac-Toe experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts_alphazero.alphazero import AlphaZeroMCTS, SimplePolicyValueNetwork, train_alphazero
from mcts_alphazero.mcts import PureMCTS
from mcts_alphazero.players import MCTSPlayer, RandomPlayer, play_game
from mcts_alphazero.tictactoe import TicTacToeState


class AlphaZeroPlayer:
    def __init__(self, mcts: AlphaZeroMCTS) -> None:
        self.mcts = mcts

    def choose_action(self, state: TicTacToeState) -> int:
        return self.mcts.choose_action(state)

    def update_with_action(self, action: int) -> None:
        self.mcts.update_with_action(action)


def evaluate_vs_random(kind: str, budgets: list[int], games: int, seed: int) -> list[dict[str, float]]:
    rows = []
    network = SimplePolicyValueNetwork(seed=seed)
    if kind == "alphazero":
        train_alphazero(network, iterations=1, games_per_iteration=3, simulations=15, epochs=5, seed=seed)

    for budget in budgets:
        wins = draws = losses = 0
        start = time.perf_counter()
        for game_idx in range(games):
            if kind == "pure":
                player = MCTSPlayer(PureMCTS(budget, tree_reuse=True, seed=seed + game_idx))
            else:
                mcts = AlphaZeroMCTS(network, budget, tree_reuse=True, seed=seed + game_idx)
                player = AlphaZeroPlayer(mcts)
            result = play_game(player, RandomPlayer(seed + 1000 + game_idx))
            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1
            else:
                losses += 1
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "kind": kind,
                "budget": budget,
                "games": games,
                "win_rate": wins / games,
                "draw_rate": draws / games,
                "loss_rate": losses / games,
                "avg_game_seconds": elapsed / games,
            }
        )
    return rows


def allocation_snapshot(budget: int, seed: int) -> dict[str, object]:
    state = TicTacToeState.from_rows(["X..", ".O.", "..."], player=1)
    network = SimplePolicyValueNetwork(seed=seed)
    train_alphazero(network, iterations=1, games_per_iteration=3, simulations=15, epochs=5, seed=seed)

    pure = PureMCTS(budget, tree_reuse=False, seed=seed)
    pure.search(state)
    az = AlphaZeroMCTS(network, budget, tree_reuse=False, seed=seed)
    az.search(state)

    priors, _ = network.predict(state)
    return {
        "state": state.render(),
        "budget": budget,
        "pure_visits": pure.root_visit_distribution(),
        "alphazero_visits": az.visit_distribution(),
        "alphazero_priors": priors,
    }


def write_plots(results: list[dict[str, float]], allocation: dict[str, object], output_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    fig, ax = plt.subplots()
    for kind in ("pure", "alphazero"):
        rows = [row for row in results if row["kind"] == kind]
        ax.plot([row["budget"] for row in rows], [row["win_rate"] for row in rows], marker="o", label=kind)
    ax.set_xscale("log")
    ax.set_xlabel("Simulation budget")
    ax.set_ylabel("Win rate vs RandomPlayer")
    ax.legend()
    path = output_dir / "performance_vs_budget.png"
    fig.savefig(path, bbox_inches="tight")
    generated.append(str(path))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for axis, key, title in (
        (axes[0], "pure_visits", "UCT visit allocation"),
        (axes[1], "alphazero_visits", "PUCT visit allocation"),
    ):
        values = allocation[key]
        grid = [values[i : i + 3] for i in (0, 3, 6)]
        image = axis.imshow(grid, vmin=0.0, vmax=max(max(values), 1e-9), cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        for idx, value in enumerate(values):
            axis.text(idx % 3, idx // 3, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, fraction=0.046)
    path = output_dir / "visit_allocation.png"
    fig.savefig(path, bbox_inches="tight")
    generated.append(str(path))
    plt.close(fig)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    results.extend(evaluate_vs_random("pure", args.budgets, args.games, args.seed))
    results.extend(evaluate_vs_random("alphazero", args.budgets, args.games, args.seed))
    allocation = allocation_snapshot(max(args.budgets), args.seed)

    (args.output_dir / "comparison_results.json").write_text(
        json.dumps({"performance": results, "allocation": allocation}, indent=2)
    )
    plots = write_plots(results, allocation, args.output_dir)
    print(json.dumps({"results": str(args.output_dir / "comparison_results.json"), "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
