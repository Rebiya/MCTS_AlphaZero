# MCTS AlphaZero

Educational reference material for Monte Carlo Tree Search (MCTS) and the
AlphaZero-style extension that combines tree search with policy and value
networks.

## Contents

- [Reference notes](docs/mcts-alphazero-reference.md): practical lookup guide
  covering classic MCTS, UCT, AlphaZero, PUCT, action masking, tree reuse,
  self-play training, and zero-sum value handling.
- [Theory guide](docs/theory.md): intermediate explanation of the progression
  from classical MCTS to AlphaZero and how the code maps across both systems.

## Implementation

- `mcts_alphazero/tictactoe.py`: shared immutable Tic-Tac-Toe environment.
- `mcts_alphazero/mcts.py`: pure MCTS with explicit selection, expansion,
  simulation, and backpropagation phases.
- `mcts_alphazero/alphazero.py`: small AlphaZero-style extension with PUCT,
  policy masking, neural value evaluation, self-play examples, and training.
- `experiments/run_experiments.py`: lightweight comparison experiments and
  Matplotlib plots.
- `tests/`: focused unit tests for the game, MCTS, tree reuse, and AlphaZero
  behavior.

## Quick Start

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Install the optional plotting dependency:

```bash
python -m pip install -r requirements.txt
```

Run a small comparison:

```bash
python experiments/run_experiments.py --budgets 10 25 50 --games 4 --seed 11
```

The experiment writes actual run data to `results/comparison_results.json`.
When Matplotlib is installed, it also writes:

- `results/performance_vs_budget.png`
- `results/visit_allocation.png`

## Learning Path

1. Understand the four phases of classic MCTS: selection, expansion,
   simulation, and backpropagation.
2. Derive and test the UCT formula on a tiny tree.
3. Replace random rollouts with a value estimate, as AlphaZero does.
4. Compare UCT and PUCT selection behavior.
5. Study self-play training and the AlphaZero loss.
6. Implement a minimal MCTS, then add PUCT and a dummy policy/value model.

## Core Formulas

Classic UCT:

```text
Q + c * sqrt(ln(N_parent) / N_child)
```

AlphaZero PUCT:

```text
Q + c * P * sqrt(N_parent) / (1 + N_child)
```

AlphaZero loss:

```text
(z - v)^2 - pi^T ln(p) + c||theta||^2
```

Zero-sum negamax value flip:

```text
value_for_me = -value_for_opponent
```
