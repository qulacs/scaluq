import unittest

import numpy as np
import scaluq as s


class APIRegressionTest(unittest.TestCase):
    def test_optimized_circuit_uses_requested_space(self):
        spaces = ["default", "host_serial"]
        if s.get_default_execution_space() != "host":
            spaces.append("host")
        for precision in ("f64", "f32"):
            if not s.precision_available(precision):
                continue
            for space in spaces:
                with self.subTest(precision=precision, space=space):
                    c = s.Circuit(precision=precision)
                    c.add_gate(s.gate.H(0, precision=precision))
                    c.add_gate(s.gate.CX(0, 1, precision=precision))
                    if space == "default":
                        c.optimize()
                    else:
                        c.optimize(space=space)
                    state = s.StateVector(2, precision=precision, space=space)
                    c.update_quantum_state(state)
                    np.testing.assert_allclose(state.get_amplitudes(), [2**-0.5, 0, 0, 2**-0.5], atol=2e-6)
        with self.assertRaises(ValueError):
            s.Circuit().optimize(space="invalid")

    def test_density_matrix_gate_dispatch(self):
        spaces = ["default", "host_serial"]
        if s.get_default_execution_space() != "host":
            spaces.append("host")
        for space in spaces:
            for precision in ("f64", "f32"):
                if not s.precision_available(precision):
                    continue
                with self.subTest(space=space, precision=precision):
                    vec = np.array([1, 2j, -3, 4j]) / np.sqrt(30)
                    expected = np.outer(vec, vec.conj())
                    density = s.DensityMatrix(2, precision=precision, space=space)
                    density.load(expected, is_hermitian=True)
                    h = s.gate.H(0, precision=precision)
                    s.HGate(h).update_quantum_state(density, seed=3)
                    u = np.kron(np.eye(2), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
                    expected = u @ expected @ u.conj().T
                    s.gate.CX(0, 1, precision=precision).update_quantum_state(
                        density, classical_register=s.ClassicalRegister(0), seed=4)
                    u = np.eye(4)[[0, 3, 2, 1]]
                    expected = u @ expected @ u.conj().T
                    np.testing.assert_allclose(density.get_matrix(), expected, atol=2e-6)
                    with self.assertRaises(RuntimeError):
                        h.update_quantum_state(density, classical_register=s.ClassicalRegisterBatched(1, 1))
                    with self.assertRaises(TypeError):
                        s.gate.ParamRX(0, precision=precision).update_quantum_state(density, 0.5)

    def test_factory_instance_checks(self):
        for precision in ("f64", "f32"):
            if not s.precision_available(precision):
                continue
            for factory, args in ((s.StateVector, (1,)), (s.StateVectorBatched, (2, 1)),
                                  (s.DensityMatrix, (1,)), (s.Circuit, ()),
                                  (s.PauliOperator, ()), (s.Operator, ([],)),
                                  (s.OperatorBatched, ([],))):
                with self.subTest(factory=factory, precision=precision):
                    obj = factory(*args, precision=precision)
                    self.assertIsInstance(obj, factory)
                    self.assertTrue(issubclass(type(obj), factory))
                    self.assertNotIsInstance(object(), factory)
        self.assertNotIsInstance(s.StateVector(1), s.DensityMatrix)

    def test_dense_inverse_without_unitary_hint(self):
        matrix = np.array([[1 + 1j, 1j], [0, 2]], dtype=complex)
        gate = s.gate.DenseMatrix([0], matrix)
        inverse = gate.get_inverse()
        np.testing.assert_allclose(inverse.get_matrix() @ matrix, np.eye(2), atol=1e-12)
        with self.assertRaises(RuntimeError):
            s.gate.DenseMatrix([0], np.array([[1, 2], [2, 4]], dtype=complex)).get_inverse()


if __name__ == "__main__":
    unittest.main()
