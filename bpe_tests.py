__import__("unittest").util._MAX_LENGTH=10000
import unittest
import bpe

class BPeTests(unittest.TestCase):
    def test_train_one_round(self):
        text = "The fox jumped over the fence!"
        encoded, tokens, token_ids, steps = bpe.train(text, k=1)
        self.assertEqual(encoded, ['T', 'he', ' ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 'e', 'd', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'he', ' ', 'f', 'e', 'n', 'c', 'e', '!'])
        self.assertEqual(tokens, {0: 'T', 1: 'h', 2: 'e', 3: ' ', 4: 'f', 5: 'o', 6: 'x', 7: ' ', 8: 'j', 9: 'u', 10: 'm', 11: 'p', 12: 'e', 13: 'd', 14: ' ', 15: 'o', 16: 'v', 17: 'e', 18: 'r', 19: ' ', 20: 't', 21: 'h', 22: 'e', 23: ' ', 24: 'f', 25: 'e', 26: 'n', 27: 'c', 28: 'e', 29: '!', 30: 'he'})
        self.assertEqual(token_ids, {v: k for k, v in tokens.items()})
        self.assertEqual(steps, [('h', 'e')])
        self.assertLess(len(encoded), len(text))

    def test_train_two_rounds(self):
        text = "The fox jumped over the fence!"
        encoded, tokens, token_ids, steps = bpe.train(text, k=2)
        self.assertEqual(encoded, ['T', 'he ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 'e', 'd', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'he ', 'f', 'e', 'n', 'c', 'e', '!'])
        self.assertEqual(tokens, {0: 'T', 1: 'h', 2: 'e', 3: ' ', 4: 'f', 5:
                                  'o', 6: 'x', 7: ' ', 8: 'j', 9: 'u', 10: 'm',
                                  11: 'p', 12: 'e', 13: 'd', 14: ' ', 15: 'o',
                                  16: 'v', 17: 'e', 18: 'r', 19: ' ', 20: 't',
                                  21: 'h', 22: 'e', 23: ' ', 24: 'f', 25: 'e',
                                  26: 'n', 27: 'c', 28: 'e', 29: '!', 30: 'he',
                                  31: 'he '})
        self.assertEqual(token_ids, {v: k for k, v in tokens.items()})
        self.assertEqual(steps, [('h', 'e'), ('he', ' ')])
        self.assertLess(len(encoded), len(text))

if __name__ == '__main__':
    unittest.main()
