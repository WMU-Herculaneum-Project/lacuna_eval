import unittest
import math
from lacuna_eval.metrics import relevance, discount, dcg, ndcg, in_top_k, top_1


class TestEvaluator(unittest.TestCase):

    def test_relevance(self):
        self.assertEqual(relevance("a", {"a", "b", "c"}), 1.0)
        self.assertEqual(relevance("d", {"a", "b", "c"}), 0.0)

    def test_discount(self):
        self.assertAlmostEqual(discount(1), math.log(2, 2))
        self.assertAlmostEqual(discount(2), math.log(2, 3))

    def test_dcg(self):
        self.assertAlmostEqual(
            dcg(["a", "b", "c"], {"a", "b", "c"}),
            1.0 / math.log(2, 2) + 1.0 / math.log(2, 3) + 1.0 / math.log(2, 4),
        )
        self.assertAlmostEqual(
            dcg(["a", "d", "c"], {"a", "b", "c"}, 2), 1.0 / math.log(2, 2)
        )

    def test_ndcg(self):
        self.assertAlmostEqual(ndcg(["a", "b", "c"], {"a", "b", "c"}), 1.0)
        # self.assertAlmostEqual(
        #    ndcg(["a", "d", "c"], {"a", "b", "c"}, 2),
        #    (1.0 / math.log(2, 2)) / (1.0 / math.log(2, 2) + 1.0 / math.log(2, 3)),
        # )

    def test_in_top_k(self):
        self.assertEqual(in_top_k(["a", "b", "c"], {"a", "b", "c"}, 2), 1.0)
        self.assertEqual(in_top_k(["d", "e", "f"], {"a", "b", "c"}, 2), 0.0)

    def test_top_1(self):
        self.assertEqual(top_1(["a", "b", "c"], {"a", "b", "c"}), 1.0)
        self.assertEqual(top_1(["d", "e", "f"], {"a", "b", "c"}), 0.0)


if __name__ == "__main__":
    unittest.main()
