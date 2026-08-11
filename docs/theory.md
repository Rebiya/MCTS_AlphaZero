# From Classical MCTS to AlphaZero

This project implements two Tic-Tac-Toe search systems on the same environment:
pure MCTS with UCT and random rollouts, and an AlphaZero-style extension with
PUCT, policy priors, value evaluation, and self-play training examples.

## Classical MCTS

Classical Monte Carlo Tree Search is an online search algorithm. It does not
try to enumerate the full game tree. Instead, it grows a partial tree where
search effort follows the most useful evidence found so far.

Each simulation has four explicit phases in `PureMCTS`:

| Phase | Code | Role |
| --- | --- | --- |
| Selection | `select` | Walk from the root through fully expanded nodes using UCT. |
| Expansion | `expand` | Add exactly one untried legal action as a new child. |
| Simulation | `simulate` | Play random legal moves until the game ends. |
| Backpropagation | `backpropagate` | Update visits and values along the path. |

UCT balances exploitation and exploration:

```text
UCT(s,a) = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
```

`Q(s,a)` estimates how good an action has been so far. The exploration term is
larger for actions with fewer visits, so the search keeps checking alternatives
instead of committing too early. In this code, unvisited children receive
infinite UCT priority, which makes each legal move get explored.

Tic-Tac-Toe is zero-sum, so values alternate perspective each ply. Nodes store
value from the perspective of the player to move at that node. When a value is
backpropagated to the parent, the sign flips. During selection, a child value is
negated before the parent uses it, because a good state for the opponent is bad
for the current player.

The default UCT exploration constant is `sqrt(2)`, the common UCB1 value. It is
a reasonable educational default for Tic-Tac-Toe, but the implementation exposes
it as `exploration_constant` for experiments.

## AlphaGo, AlphaGo Zero, and AlphaZero

Earlier AlphaGo combined several learned and search components: policy networks
to suggest moves, a value network to evaluate positions, and rollout policies to
finish games quickly. It also used human expert games before self-play became a
major source of improvement.

AlphaGo Zero simplified the design around self-play and a combined
policy-value network. Instead of using random or fast rollout evaluation as the
main leaf estimate, the network predicted both a move distribution and a value
for the current state.

AlphaZero generalized the same idea across games such as Chess, Shogi, and Go:
learn from self-play, guide MCTS with a policy prior, and train the network
toward the stronger policy produced by search.

## Policy-Value Network

The AlphaZero-style extension uses:

```text
state
  |
  v
policy-value network
  |- P(s,a)
  `- V(s)
```

The policy output `P(a | s)` says which legal actions the model currently
believes are promising. The value output `V(s)` estimates how favorable the
state is for the player to move.

In this repository the network is intentionally tiny and dependency-light: a
linear policy head and a linear value head trained with manual gradients. It is
not meant to be a strong neural architecture. It exists to make the algorithmic
replacement concrete: policy priors guide selection, and value estimates replace
random terminal rollouts.

## Actor-Critic Analogy

The policy head is actor-like because it proposes actions. The value head is
critic-like because it evaluates states.

That analogy is useful architecturally, but AlphaZero should not be reduced to a
standard actor-critic algorithm. The key policy improvement mechanism here is
MCTS: the network's policy is used to guide search, search produces a stronger
visit distribution, and training moves the network toward that distribution.

## UCT vs PUCT

Classical UCT:

```text
UCT(s,a) = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
```

AlphaZero-style PUCT:

```text
PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

The conceptual change is `P(s,a)`: PUCT explores according to both uncertainty
and the network's prior belief. A high-prior move receives more early search
effort. A low-prior move is not impossible, but it must compete against moves
the network already considers plausible.

The formulas also differ in how parent visits appear. UCT commonly uses
`ln(N(s))`, while PUCT often uses `sqrt(N(s))` scaled by a learned prior. Exact
variants differ by implementation, but the important distinction is that UCT is
driven by search statistics alone, while PUCT mixes search statistics with a
policy prior. The `1 + N(s,a)` denominator in PUCT keeps unvisited children
finite rather than assigning infinite priority.

## Self-Play Training

One training iteration looks like:

```text
network
   |
   v
self-play
   |
   v
MCTS + PUCT
   |
   v
pi
   |
   v
game outcome z
   |
   v
(s, pi, z)
   |
   v
network training
   |
   v
improved network
```

`pi` is the MCTS visit distribution at a state. It is a stronger target than the
network's raw prediction because it includes extra computation from search. In
short:

```text
network intuition -> search -> better decision distribution -> training -> better intuition
```

The implemented loss is the AlphaZero-style combination:

```text
L = (z - v)^2 - pi^T log(p) + c||theta||^2
```

The code computes mean-squared value error for `(z - v)^2`, cross-entropy for
`-pi^T log(p)`, and weight decay for the regularization term.

## Code Mapping

| Pure MCTS | AlphaZero-style MCTS |
| --- | --- |
| `PureMCTS` | `AlphaZeroMCTS` |
| UCT | PUCT |
| random rollout in `simulate` | value-network evaluation in `evaluate_and_expand` |
| no learned prior | policy prior `P(s,a)` on each child |
| fixed search heuristic | learned search guidance |
| no training | self-play plus `(s, pi, z)` training |
| search statistics | search statistics plus network predictions |

Both implementations reuse `TicTacToeState`, legal action handling, terminal
detection, zero-sum result conventions, and tree reuse mechanics.
