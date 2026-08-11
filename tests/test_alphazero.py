import unittest

from mcts_alphazero.alphazero import (
    AlphaZeroMCTS,
    AlphaZeroNode,
    SimplePolicyValueNetwork,
    mask_policy,
    self_play_game,
)
from mcts_alphazero.tictactoe import TicTacToeState


class AlphaZeroTests(unittest.TestCase):
    def test_policy_and_value_shapes(self):
        network = SimplePolicyValueNetwork(seed=1)
        policy, value = network.predict(TicTacToeState())
        self.assertEqual(len(policy), 9)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_mask_policy_removes_illegal_actions_and_normalizes(self):
        masked = mask_policy([1.0 / 9.0] * 9, [0, 4, 8])
        self.assertAlmostEqual(sum(masked), 1.0)
        self.assertEqual(masked[1], 0.0)
        self.assertEqual(masked[0], 1.0 / 3.0)

    def test_puct_uses_policy_prior(self):
        network = SimplePolicyValueNetwork(seed=2)
        mcts = AlphaZeroMCTS(network, c_puct=2.0)
        parent = AlphaZeroNode(TicTacToeState(), visits=10)
        low = AlphaZeroNode(parent.state.apply(0), prior=0.1, parent=parent, visits=1, total_value=0.0)
        high = AlphaZeroNode(parent.state.apply(1), prior=0.8, parent=parent, visits=1, total_value=0.0)
        self.assertGreater(mcts.puct_score(parent, high), mcts.puct_score(parent, low))

    def test_visit_distribution_is_normalized(self):
        network = SimplePolicyValueNetwork(seed=3)
        mcts = AlphaZeroMCTS(network, simulation_budget=15, seed=3)
        mcts.search(TicTacToeState())
        pi = mcts.visit_distribution()
        self.assertEqual(len(pi), 9)
        self.assertAlmostEqual(sum(pi), 1.0)

    def test_self_play_produces_valid_examples(self):
        network = SimplePolicyValueNetwork(seed=4)
        examples = self_play_game(network, simulations=5, seed=4)
        self.assertGreater(len(examples), 0)
        for state, pi, z in examples:
            self.assertIsInstance(state, TicTacToeState)
            self.assertEqual(len(pi), 9)
            self.assertAlmostEqual(sum(pi), 1.0)
            self.assertIn(z, (-1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
