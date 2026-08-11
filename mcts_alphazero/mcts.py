"""Classic Monte Carlo Tree Search with UCT and random rollouts."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .tictactoe import TicTacToeState


@dataclass
class MCTSNode:
    """Tree node storing values from the perspective of the player to move."""

    state: TicTacToeState
    parent: "MCTSNode | None" = None
    action: int | None = None
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = self.state.legal_actions()

    @property
    def player_to_move(self) -> int:
        return self.state.player

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions


class PureMCTS:
    """Pure MCTS using UCT selection and random terminal rollouts."""

    def __init__(
        self,
        simulation_budget: int = 100,
        exploration_constant: float = math.sqrt(2.0),
        tree_reuse: bool = True,
        seed: int | None = None,
    ) -> None:
        self.simulation_budget = simulation_budget
        self.exploration_constant = exploration_constant
        self.tree_reuse = tree_reuse
        self.random = random.Random(seed)
        self.root: MCTSNode | None = None

    def search(self, state: TicTacToeState) -> MCTSNode:
        """Run the configured number of simulations and return the root node."""

        self._ensure_root(state)
        assert self.root is not None
        for _ in range(self.simulation_budget):
            node = self.select(self.root)
            if not node.state.is_terminal():
                node = self.expand(node)
            value = self.simulate(node.state)
            self.backpropagate(node, value)
        return self.root

    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: descend through fully expanded nodes by UCT."""

        while not node.state.is_terminal() and node.is_fully_expanded() and node.children:
            node = self._best_uct_child(node)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add exactly one legal untried child."""

        action = self.random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child = MCTSNode(state=node.state.apply(action), parent=node, action=action)
        node.children[action] = child
        return child

    def simulate(self, state: TicTacToeState) -> float:
        """Simulation phase: random rollout to terminal result."""

        rollout_state = state
        starting_player = state.player
        while not rollout_state.is_terminal():
            action = self.random.choice(rollout_state.legal_actions())
            rollout_state = rollout_state.apply(action)
        return rollout_state.result_for_player(starting_player)

    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation phase with zero-sum sign flips at each ply."""

        current: MCTSNode | None = node
        current_value = value
        while current is not None:
            current.visits += 1
            current.total_value += current_value
            current_value = -current_value
            current = current.parent

    def choose_action(self, state: TicTacToeState) -> int:
        """Search and choose the legal action with the most root visits."""

        root = self.search(state)
        if not root.children:
            legal = state.legal_actions()
            if not legal:
                raise ValueError("no legal actions available")
            return self.random.choice(legal)
        return max(root.children.items(), key=lambda item: item[1].visits)[0]

    def update_with_action(self, action: int) -> None:
        """Reuse the matching child as the next root after a real move."""

        if not self.tree_reuse or self.root is None or action not in self.root.children:
            self.root = None
            return
        self.root = self.root.children[action]
        self.root.parent = None
        self.root.action = None

    def root_visit_distribution(self) -> list[float]:
        """Return a length-9 visit distribution over actions from the current root."""

        if self.root is None:
            return [0.0] * 9
        total = sum(child.visits for child in self.root.children.values())
        if total == 0:
            return [0.0] * 9
        visits = [0.0] * 9
        for action, child in self.root.children.items():
            visits[action] = child.visits / total
        return visits

    def _ensure_root(self, state: TicTacToeState) -> None:
        if self.tree_reuse and self.root is not None and self.root.state == state:
            return
        self.root = MCTSNode(state)

    def _best_uct_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children.values(), key=lambda child: self.uct_score(node, child))

    def uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """UCT score for choosing `child` from `parent`.

        Child `q` is stored from the child player perspective, so the parent
        sees that value negated.
        """

        if child.visits == 0:
            return math.inf
        exploitation = -child.q
        exploration = self.exploration_constant * math.sqrt(
            math.log(max(parent.visits, 1)) / child.visits
        )
        return exploitation + exploration
