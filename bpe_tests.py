__import__("unittest").util._MAX_LENGTH=10000
import unittest
import bpe

class BPeTests(unittest.TestCase):
    def test_train_one_round(self):
        text = "The fox jumped over the fence!"
        encoded, tokens, token_ids, steps = bpe.train(text, k=1)
        self.assertEqual(encoded, ['T', 'he', ' ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 'e', 'd', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'he', ' ', 'f', 'e', 'n', 'c', 'e', '!'])
        self.assertEqual(tokens, {0: 'T', 1: 'h', 2: 'e', 3: ' ', 4: 'f', 5:
                                  'o', 6: 'x', 7: 'j', 8: 'u', 9: 'm', 10: 'p',
                                  11: 'd', 12: 'v', 13: 'r', 14: 't', 15: 'n',
                                  16: 'c', 17: '!', 18: 'he'})
        self.assertEqual(token_ids, {v: k for k, v in tokens.items()})
        self.assertEqual(steps, [('h', 'e')])
        self.assertLess(len(encoded), len(text))

    def test_train_two_rounds(self):
        text = "The fox jumped over the fence!"
        encoded, tokens, token_ids, steps = bpe.train(text, k=2)
        self.assertEqual(encoded, ['T', 'he ', 'f', 'o', 'x', ' ', 'j', 'u', 'm', 'p', 'e', 'd', ' ', 'o', 'v', 'e', 'r', ' ', 't', 'he ', 'f', 'e', 'n', 'c', 'e', '!'])
        self.assertEqual(tokens, {0: 'T', 1: 'h', 2: 'e', 3: ' ', 4: 'f', 5:
                                  'o', 6: 'x', 7: 'j', 8: 'u', 9: 'm', 10: 'p',
                                  11: 'd', 12: 'v', 13: 'r', 14: 't', 15: 'n',
                                  16: 'c', 17: '!', 18: 'he', 19: 'he '})
        self.assertEqual(token_ids, {v: k for k, v in tokens.items()})
        self.assertEqual(steps, [('h', 'e'), ('he', ' ')])
        self.assertLess(len(encoded), len(text))

    def test_train_more_rounds(self):
        text = "Byte Pair Encoding (BPE) is a data compression technique that iteratively merges the most frequent pair of consecutive bytes (or characters) in a text or data sequence into a single, new symbol. The process is repeated until a specified number of merges is reached or no more frequent pairs remain BPE BPE"
        encoded, tokens, token_ids, steps = bpe.train(text, k=20)
        self.assertEqual(encoded, ['B', 'y', 'te', ' ', 'P', 'ai', 'r ', 'E', 'n', 'co', 'd', 'in', 'g', ' ', '(', 'BPE', ')', ' ', 'is ', 'a ', 'd', 'at', 'a ', 'co', 'm', 'p', 're', 's', 's', 'i', 'o', 'n', ' ', 'te', 'ch', 'n', 'i', 'qu', 'e ', 't', 'h', 'a', 't ', 'i', 'te', 'r', 'at', 'i', 'v', 'e', 'l', 'y', ' m', 'e', 'r', 'g', 'e', 's ', 't', 'h', 'e ', 'm', 'o', 's', 't ', 'f', 're', 'quen', 't ', 'p', 'ai', 'r ', 'o', 'f', ' ', 'co', 'n', 's', 'e', 'c', 'u', 't', 'i', 'v', 'e ', 'b', 'y', 'te', 's ', '(', 'or ', 'ch', 'a', 'r', 'a', 'c', 'te', 'r', 's', ')', ' ', 'in', ' ', 'a ', 'te', 'x', 't ', 'or ', 'd', 'at', 'a ', 's', 'e', 'quen', 'c', 'e ', 'in', 't', 'o', ' ', 'a ', 's', 'in', 'g', 'l', 'e', ',', ' ', 'n', 'e', 'w', ' ', 's', 'y', 'm', 'b', 'o', 'l', '.', ' ', 'T', 'h', 'e ', 'p', 'r', 'o', 'c', 'e', 's', 's ', 'is ', 're', 'p', 'e', 'a', 'te', 'd', ' ', 'u', 'n', 't', 'i', 'l', ' ', 'a ', 's', 'p', 'e', 'c', 'i', 'f', 'i', 'e', 'd', ' ', 'n', 'u', 'm', 'b', 'e', 'r ', 'o', 'f', ' m', 'e', 'r', 'g', 'e', 's ', 'is ', 're', 'a', 'ch', 'e', 'd', ' ', 'or ', 'n', 'o', ' m', 'o', 're', ' ', 'f', 're', 'quen', 't ', 'p', 'ai', 'r', 's ', 're', 'm', 'a', 'in', ' ', 'BPE', ' ', 'BPE'])
        self.assertEqual(tokens, {0: 'B', 1: 'y', 2: 't', 3: 'e', 4: ' ', 5:
                                  'P', 6: 'a', 7: 'i', 8: 'r', 9: 'E', 10: 'n',
                                  11: 'c', 12: 'o', 13: 'd', 14: 'g', 15: '(',
                                  16: ')', 17: 's', 18: 'm', 19: 'p', 20: 'h',
                                  21: 'q', 22: 'u', 23: 'v', 24: 'l', 25: 'f',
                                  26: 'b', 27: 'x', 28: ',', 29: 'w', 30: '.',
                                  31: 'T', 32: 's ', 33: 'te', 34: 're', 35: 'r ', 36: 'a ', 37: 'in', 38: 'e ', 39: 't ',
                                  40: 'qu', 41: 'ai', 42: 'co', 43: 'BP', 44:
                                  'BPE', 45: 'is ', 46: 'at', 47: 'ch', 48: ' m', 49: 'que', 50: 'quen', 51: 'or '})
        self.assertEqual(token_ids, {v: k for k, v in tokens.items()})
        self.assertEqual(steps, [('s', ' '), ('t', 'e'), ('r', 'e'), ('r', ' '), ('a', ' '), ('i', 'n'), ('e', ' '), ('t', ' '), ('q', 'u'), ('a', 'i'), ('c', 'o'), ('B', 'P'), ('BP', 'E'), ('i', 's '), ('a', 't'), ('c', 'h'), (' ', 'm'), ('qu', 'e'), ('que', 'n'), ('o', 'r ')])
        self.assertLess(len(encoded), len(text))

    def test_encode(self):
        text = "the fox jumped over the fence"
        _, tokens, token_ids, steps = bpe.train(text, k=3)
        encoded = bpe.encode(token_ids, steps, "the")
        self.assertEqual([tokens[x] for x in encoded], ["the"])

if __name__ == '__main__':
    unittest.main()
