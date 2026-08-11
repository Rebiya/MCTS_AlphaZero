import unittest

from mcts_alphazero.tictactoe import TicTacToeState


class TicTacToeTests(unittest.TestCase):
    def test_legal_actions_and_apply(self):
        state = TicTacToeState()
        self.assertEqual(state.legal_actions(), list(range(9)))
        next_state = state.apply(4)
        self.assertEqual(next_state.board[4], 1)
        self.assertEqual(next_state.player, -1)
        self.assertNotIn(4, next_state.legal_actions())

    def test_winner_detection(self):
        state = TicTacToeState.from_rows(["XXX", "OO.", "..."], player=-1)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.winner(), 1)
        self.assertEqual(state.result_for_player(1), 1.0)
        self.assertEqual(state.result_for_player(-1), -1.0)

    def test_draw_detection(self):
        state = TicTacToeState.from_rows(["XOX", "XXO", "OXO"], player=1)
        self.assertTrue(state.is_terminal())
        self.assertEqual(state.winner(), 0)
        self.assertEqual(state.result_for_player(1), 0.0)


if __name__ == "__main__":
    unittest.main()
