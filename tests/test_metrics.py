import math
import unittest

from lacuna_eval.metrics import dcg
from lacuna_eval.metrics import discount
from lacuna_eval.metrics import evaluate
from lacuna_eval.metrics import in_top_k
from lacuna_eval.metrics import ndcg
from lacuna_eval.metrics import relevance
from lacuna_eval.metrics import top_1


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

    def test_evaluate(self):
        results = evaluate(
            [["a", "b"], ["d", "a"]],
            [{"a"}, {"a"}],
            k=2,
        )

        self.assertEqual(results["count"], 2)
        expected_mean_dcg = (1.0 + 1.0 / math.log(2, 3)) / 2
        self.assertAlmostEqual(results["mean_dcg"], expected_mean_dcg)
        self.assertAlmostEqual(results["mean_ndcg"], 0.5)
        self.assertAlmostEqual(results["top_1_accuracy"], 0.5)
        self.assertEqual(results["top_k_accuracy"], 1.0)

    def test_evaluate_rejects_mismatched_pairs(self):
        with self.assertRaises(ValueError):
            evaluate([["a"]], [])

    def test_evaluate_empty_batch(self):
        self.assertEqual(
            evaluate([], [], k=2),
            {
                "count": 0,
                "mean_dcg": 0.0,
                "mean_ndcg": 0.0,
                "top_1_accuracy": 0.0,
                "top_k_accuracy": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
