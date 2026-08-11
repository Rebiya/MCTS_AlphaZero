import math
import unittest

from mcts_alphazero.mcts import MCTSNode, PureMCTS
from mcts_alphazero.tictactoe import TicTacToeState


class PureMCTSTests(unittest.TestCase):
    def test_uct_prefers_parent_perspective_value(self):
        state = TicTacToeState()
        parent = MCTSNode(state)
        child_a = MCTSNode(state.apply(0), parent=parent, action=0, visits=10, total_value=-8.0)
        child_b = MCTSNode(state.apply(1), parent=parent, action=1, visits=10, total_value=2.0)
        parent.children = {0: child_a, 1: child_b}
        parent.visits = 20
        mcts = PureMCTS(exploration_constant=0.0)
        self.assertGreater(mcts.uct_score(parent, child_a), mcts.uct_score(parent, child_b))

    def test_unvisited_child_has_infinite_uct(self):
        parent = MCTSNode(TicTacToeState(), visits=1)
        child = MCTSNode(parent.state.apply(0), parent=parent, visits=0)
        self.assertEqual(PureMCTS().uct_score(parent, child), math.inf)

    def test_expansion_only_creates_legal_action(self):
        state = TicTacToeState.from_rows(["X..", "...", "..."], player=-1)
        root = MCTSNode(state)
        child = PureMCTS(seed=1).expand(root)
        self.assertIn(child.action, range(1, 9))
        self.assertNotEqual(child.action, 0)

    def test_simulation_returns_valid_outcome(self):
        value = PureMCTS(seed=2).simulate(TicTacToeState())
        self.assertIn(value, (-1.0, 0.0, 1.0))

    def test_backpropagation_flips_zero_sum_perspective(self):
        root = MCTSNode(TicTacToeState())
        child = MCTSNode(root.state.apply(0), parent=root, action=0)
        PureMCTS().backpropagate(child, value=1.0)
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.total_value, 1.0)
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.total_value, -1.0)

    def test_tree_reuse_preserves_selected_subtree(self):
        mcts = PureMCTS(simulation_budget=20, tree_reuse=True, seed=3)
        state = TicTacToeState()
        root = mcts.search(state)
        action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        child = root.children[action]
        old_visits = child.visits
        mcts.update_with_action(action)
        self.assertIs(mcts.root, child)
        self.assertIsNone(mcts.root.parent)
        self.assertIsNone(mcts.root.action)
        self.assertEqual(mcts.root.visits, old_visits)


if __name__ == "__main__":
    unittest.main()
