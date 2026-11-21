import unittest

import numpy as np
import scaluq


class PauliMatrixTripletsTest(unittest.TestCase):
    def test_triplets_reconstruct_the_matrix_without_the_coefficient(self):
        for precision in scaluq.get_available_precisions():
            for pauli_string in ["", "X 0", "Y 0", "Z 0", "X 0 Y 2 Z 4"]:
                for coefficient in [0, 2j]:
                    with self.subTest(
                        precision=precision,
                        pauli=pauli_string,
                        coefficient=coefficient,
                    ):
                        pauli = scaluq.PauliOperator(
                            pauli_string, coefficient, precision=precision
                        )
                        entries = pauli.get_matrix_triplets_ignoring_coef()
                        expected = pauli.get_matrix_ignoring_coef()
                        actual = np.zeros_like(expected)
                        self.assertIsInstance(entries, list)
                        for entry in entries:
                            self.assertIsInstance(entry, tuple)
                            row, column, value = entry
                            self.assertIsInstance(row, int)
                            self.assertIsInstance(column, int)
                            self.assertIsInstance(value, complex)
                            actual[row, column] += value
                        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
