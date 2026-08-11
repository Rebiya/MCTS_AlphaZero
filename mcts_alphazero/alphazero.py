"""Small dependency-light AlphaZero-style extension for Tic-Tac-Toe."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .tictactoe import TicTacToeState


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def mask_policy(policy: list[float], legal_actions: list[int]) -> list[float]:
    """Zero illegal actions and renormalize legal probabilities."""

    masked = [0.0] * 9
    total = 0.0
    for action in legal_actions:
        masked[action] = max(policy[action], 0.0)
        total += masked[action]
    if total <= 0.0 and legal_actions:
        uniform = 1.0 / len(legal_actions)
        for action in legal_actions:
            masked[action] = uniform
        return masked
    if total > 0.0:
        for action in legal_actions:
            masked[action] /= total
    return masked


class SimplePolicyValueNetwork:
    """Tiny linear policy-value model trained with manual gradients.

    This intentionally avoids a heavy neural framework so the repository remains
    runnable in minimal Python environments while still demonstrating the
    AlphaZero targets: policy distribution `pi` and value `z`.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.policy_weights = [
            [self.random.uniform(-0.05, 0.05) for _ in range(9)] for _ in range(9)
        ]
        self.policy_bias = [0.0] * 9
        self.value_weights = [self.random.uniform(-0.05, 0.05) for _ in range(9)]
        self.value_bias = 0.0

    def predict(self, state: TicTacToeState) -> tuple[list[float], float]:
        x = state.encode_for_current_player()
        logits = [
            self.policy_bias[action]
            + sum(x[i] * self.policy_weights[i][action] for i in range(9))
            for action in range(9)
        ]
        raw_value = self.value_bias + sum(x[i] * self.value_weights[i] for i in range(9))
        return softmax(logits), math.tanh(raw_value)

    def train(
        self,
        examples: list[tuple[TicTacToeState, list[float], float]],
        epochs: int = 20,
        learning_rate: float = 0.05,
        weight_decay: float = 1e-4,
    ) -> dict[str, float]:
        """Train on `(state, pi, z)` examples using cross-entropy plus MSE."""

        if not examples:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        last_policy_loss = 0.0
        last_value_loss = 0.0
        for _ in range(epochs):
            self.random.shuffle(examples)
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            for state, target_pi, target_z in examples:
                x = state.encode_for_current_player()
                policy, value = self.predict(state)

                policy_loss_sum += -sum(
                    target_pi[a] * math.log(max(policy[a], 1e-12)) for a in range(9)
                )
                value_loss_sum += (target_z - value) ** 2

                grad_logits = [policy[a] - target_pi[a] for a in range(9)]
                raw_value_grad = 2.0 * (value - target_z) * (1.0 - value * value)

                for i in range(9):
                    for action in range(9):
                        grad = x[i] * grad_logits[action]
                        grad += weight_decay * self.policy_weights[i][action]
                        self.policy_weights[i][action] -= learning_rate * grad
                    value_grad = x[i] * raw_value_grad + weight_decay * self.value_weights[i]
                    self.value_weights[i] -= learning_rate * value_grad

                for action in range(9):
                    self.policy_bias[action] -= learning_rate * grad_logits[action]
                self.value_bias -= learning_rate * raw_value_grad

            last_policy_loss = policy_loss_sum / len(examples)
            last_value_loss = value_loss_sum / len(examples)
        return {"policy_loss": last_policy_loss, "value_loss": last_value_loss}


@dataclass
class AlphaZeroNode:
    """PUCT node with a policy prior on the incoming action."""

    state: TicTacToeState
    prior: float = 1.0
    parent: "AlphaZeroNode | None" = None
    action: int | None = None
    children: dict[int, "AlphaZeroNode"] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


class AlphaZeroMCTS:
    """AlphaZero-style search using PUCT and neural leaf evaluation."""

    def __init__(
        self,
        network: SimplePolicyValueNetwork,
        simulation_budget: int = 100,
        c_puct: float = 1.5,
        tree_reuse: bool = True,
        seed: int | None = None,
    ) -> None:
        self.network = network
        self.simulation_budget = simulation_budget
        self.c_puct = c_puct
        self.tree_reuse = tree_reuse
        self.random = random.Random(seed)
        self.root: AlphaZeroNode | None = None

    def search(self, state: TicTacToeState) -> AlphaZeroNode:
        self._ensure_root(state)
        assert self.root is not None
        for _ in range(self.simulation_budget):
            node = self.select(self.root)
            value = self.evaluate_and_expand(node)
            self.backpropagate(node, value)
        return self.root

    def select(self, node: AlphaZeroNode) -> AlphaZeroNode:
        """Selection phase: descend through expanded children with PUCT."""

        while node.children and not node.state.is_terminal():
            node = max(node.children.values(), key=lambda child: self.puct_score(node, child))
        return node

    def evaluate_and_expand(self, node: AlphaZeroNode) -> float:
        """Use the policy-value network instead of a random rollout."""

        if node.state.is_terminal():
            return node.state.result_for_player(node.state.player)

        policy, value = self.network.predict(node.state)
        masked_policy = mask_policy(policy, node.state.legal_actions())
        for action in node.state.legal_actions():
            if action not in node.children:
                node.children[action] = AlphaZeroNode(
                    state=node.state.apply(action),
                    prior=masked_policy[action],
                    parent=node,
                    action=action,
                )
        return value

    def backpropagate(self, node: AlphaZeroNode, value: float) -> None:
        """Backpropagate value with negamax sign flips."""

        current: AlphaZeroNode | None = node
        current_value = value
        while current is not None:
            current.visits += 1
            current.total_value += current_value
            current_value = -current_value
            current = current.parent

    def puct_score(self, parent: AlphaZeroNode, child: AlphaZeroNode) -> float:
        """PUCT score with neural prior `P(s,a)` and parent-perspective value."""

        exploitation = -child.q
        exploration = (
            self.c_puct
            * child.prior
            * math.sqrt(max(parent.visits, 1))
            / (1 + child.visits)
        )
        return exploitation + exploration

    def visit_distribution(self, temperature: float = 1.0) -> list[float]:
        """Construct policy target pi from root visit counts."""

        if self.root is None or not self.root.children:
            return [0.0] * 9
        if temperature <= 1e-8:
            best_action = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
            pi = [0.0] * 9
            pi[best_action] = 1.0
            return pi
        counts = [0.0] * 9
        for action, child in self.root.children.items():
            counts[action] = child.visits ** (1.0 / temperature)
        total = sum(counts)
        return [count / total if total else 0.0 for count in counts]

    def choose_action(self, state: TicTacToeState, temperature: float = 0.0) -> int:
        self.search(state)
        pi = self.visit_distribution(temperature)
        if temperature <= 1e-8:
            return max(range(9), key=lambda action: pi[action])
        threshold = self.random.random()
        cumulative = 0.0
        for action, probability in enumerate(pi):
            cumulative += probability
            if threshold <= cumulative:
                return action
        return max(state.legal_actions(), key=lambda action: pi[action])

    def update_with_action(self, action: int) -> None:
        if not self.tree_reuse or self.root is None or action not in self.root.children:
            self.root = None
            return
        self.root = self.root.children[action]
        self.root.parent = None
        self.root.action = None

    def _ensure_root(self, state: TicTacToeState) -> None:
        if self.tree_reuse and self.root is not None and self.root.state == state:
            return
        self.root = AlphaZeroNode(state=state)


def self_play_game(
    network: SimplePolicyValueNetwork,
    simulations: int = 50,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    seed: int | None = None,
) -> list[tuple[TicTacToeState, list[float], float]]:
    """Generate one game's `(state, pi, z)` examples."""

    mcts = AlphaZeroMCTS(network, simulations, c_puct=c_puct, tree_reuse=True, seed=seed)
    state = TicTacToeState()
    trajectory: list[tuple[TicTacToeState, list[float], int]] = []
    while not state.is_terminal():
        mcts.search(state)
        pi = mcts.visit_distribution(temperature)
        legal = state.legal_actions()
        action = _sample_from_policy(pi, legal, mcts.random)
        trajectory.append((state, pi, state.player))
        state = state.apply(action)
        mcts.update_with_action(action)

    winner = state.winner()
    assert winner is not None
    examples = []
    for seen_state, pi, player in trajectory:
        z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        examples.append((seen_state, pi, z))
    return examples


def train_alphazero(
    network: SimplePolicyValueNetwork,
    iterations: int = 2,
    games_per_iteration: int = 4,
    simulations: int = 25,
    epochs: int = 10,
    learning_rate: float = 0.05,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Run a small educational self-play training loop."""

    rng = random.Random(seed)
    history = []
    for _ in range(iterations):
        examples: list[tuple[TicTacToeState, list[float], float]] = []
        for _ in range(games_per_iteration):
            examples.extend(self_play_game(network, simulations=simulations, seed=rng.randrange(10**9)))
        losses = network.train(examples, epochs=epochs, learning_rate=learning_rate)
        losses["examples"] = float(len(examples))
        history.append(losses)
    return history


def _sample_from_policy(policy: list[float], legal_actions: list[int], rng: random.Random) -> int:
    legal_policy = mask_policy(policy, legal_actions)
    threshold = rng.random()
    cumulative = 0.0
    for action in legal_actions:
        cumulative += legal_policy[action]
        if threshold <= cumulative:
            return action
    return legal_actions[-1]
