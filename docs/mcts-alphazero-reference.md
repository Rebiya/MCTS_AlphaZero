# Monte Carlo Tree Search & AlphaZero - Reference Notes

## 1. Foundations of Monte Carlo Tree Search

MCTS is an online, randomized heuristic search for large state spaces. It builds
an asymmetric look-ahead tree by repeating short simulations, focusing
computation on the most promising lines instead of exhaustively expanding every
child.

It works especially well for:

- Games with huge branching factors, such as Go, Chess, and Shogi
- Planning problems, such as robotics, scheduling, and synthesis routes
- Any domain where outcomes can be simulated and scored with a reward or
  win-loss signal

## 2. The Four Phases of an MCTS Iteration

Every classic MCTS iteration has four phases.

### Selection

Start at the root and traverse the tree using a selection policy, usually UCT or
PUCT, that balances exploitation of high-value moves and exploration of rarely
tried moves. Stop when reaching an unexpanded leaf or terminal state.

### Expansion

At the leaf, create exactly one new child for one untried legal action. This
lazy expansion saves memory and CPU in high-branching games.

### Simulation

Classic MCTS performs a fast random rollout from the new child to a terminal
state to get a result. AlphaZero replaces this rollout with a neural network
value head.

### Backpropagation

Propagate the outcome back up the path:

- Increment visit counts `N`
- Update total value or wins `W`
- Update mean value `Q`

Future selection then reflects the new evidence.

## 3. Multi-Armed Bandits and UCT

The exploration-exploitation tradeoff is modeled as a multi-armed bandit.

UCT, or Upper Confidence Bound applied to Trees:

```text
UCT(s, a) = Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))
```

- `Q(s, a)`: average value of action `a`
- Exploration term: grows when an action has been tried rarely
- If `N(s, a) = 0`, the exploration term is treated as infinite, so every legal
  move is tried at least once

Key behaviors:

| Behavior | What happens | Practical consequence |
| --- | --- | --- |
| Division by zero | Unvisited children get infinite priority | Forces initial exploration of all moves |
| Auto-correction | Strong moves get exploited, then their exploration bonus shrinks | Eventually revisits neglected alternatives |

## 4. AlphaZero: MCTS with Deep Learning

AlphaZero removes random rollouts and replaces them with a deep residual network
that has two heads:

- Policy head `p`: prior probabilities over actions
- Value head `v`: scalar estimate of the expected outcome in `[-1, +1]`

When selection reaches an unexpanded leaf, the network supplies:

- Priors `P(s, a)` for legal children
- A value estimate `v`

This makes search more strategic and compute-efficient in high-branching games.

### Core Self-Play Training Loop

1. Play games of self-play using MCTS guided by the current network.
2. At the end of each game, get the true outcome `z`: `-1`, `0`, or `+1`.
3. Train the network to minimize:

```text
L = (z - v)^2 - pi^T ln(p) + c||theta||^2
```

- `(z - v)^2`: value head should match the real outcome
- `-pi^T ln(p)`: policy head should match the improved visit distribution
  produced by MCTS
- `c||theta||^2`: weight decay regularizer

Over many iterations, the network's intuition and the search's calculation
co-evolve.

## 5. PUCT: Predictor + UCB

Classic UCT has a problem: unvisited actions get infinite score. AlphaZero uses
PUCT instead:

```text
PUCT(s, a) = Q(s, a) + c * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

- `P(s, a)`: prior from the policy network
- The `+1` in the denominator bounds the exploration bonus for unvisited actions
- High prior: search immediately goes deeper along that line
- Tiny prior: the action is almost ignored unless later evidence justifies it

The result is asymmetric, depth-focused search along lines the network already
considers promising.

## 6. Engineering Optimizations

### Action Masking

A fast rule engine produces a binary mask of legal moves. The policy head is
multiplied by this mask and renormalized so only legal moves receive probability
mass. The network itself can remain rule-blind.

### Tree Reuse

After a real move is played:

- The corresponding child becomes the new root
- Its subtree, visit counts, values, and priors are kept

This carries forward useful statistics and allows deeper search on later turns
with the same compute budget.

## 7. MCTS and Negamax in Zero-Sum Games

In zero-sum games, the value for the current player is the negative of the value
for the opponent.

During backpropagation, flip the sign at every level so every node stores value
from the perspective of the player about to move. Selection formulas stay
unchanged because `Q` already encodes the correct player's perspective.

Typical node contents:

- State
- Parent
- Action that led to this node
- Player to move
- Children
- Visit count `N`
- Total value or wins `W`
- Untried legal actions

## 8. Practical Takeaways

| Idea | Why it matters |
| --- | --- |
| PUCT plus policy/value network | Turns search into depth-first, intuition-guided exploration |
| Action masking plus tree reuse | Keeps search legal and reuses computation across turns |
| Self-play training loop | Continuously improves both the network and the search policy |
| Negamax sign flipping | Handles zero-sum games without changing selection formulas |

Classic MCTS with random rollouts is already powerful. AlphaZero-style MCTS,
using PUCT, neural priors, value estimation, and self-play, is what made
superhuman Chess, Shogi, and Go possible with far less domain knowledge than
traditional engines.

## 9. Quick Formula Cheat Sheet

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

Negamax idea:

```text
value_for_me = -value_for_opponent
```

