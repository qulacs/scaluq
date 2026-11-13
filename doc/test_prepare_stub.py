import ast
import unittest

from source.prepare_stub import prepare_stub


class PrepareStubTest(unittest.TestCase):
    def test_backend_types_use_local_public_classes(self):
        for space in ["default", "host", "host_serial"]:
            for precision in ["f16", "f32", "f64", "bf16"]:
                with self.subTest(space=space, precision=precision):
                    source = f'''
import scaluq.scaluq_core.{space}.{precision}
class StateVector: ...
def apply(state: scaluq.scaluq_core.{space}.{precision}.StateVector) -> scaluq_core.GateType:
    """Keep the example scaluq_core.GateType unchanged."""
    ...
'''
                    result = ast.parse(prepare_stub(source, {"GateType"}))
                    method = result.body[-1]
                    self.assertEqual(ast.unparse(method.args.args[0].annotation), "StateVector")
                    self.assertEqual(ast.unparse(method.returns), "scaluq.GateType")
                    self.assertIn("import scaluq", ast.unparse(result))
                    self.assertNotIn(f"import scaluq.scaluq_core.{space}", ast.unparse(result))
                    self.assertIn("scaluq_core.GateType", ast.get_docstring(method))

    def test_gate_stub_imports_the_public_package(self):
        source = '''
import scaluq.scaluq_core.default.f64
def create() -> scaluq.scaluq_core.default.f64.Gate: ...
'''
        result = ast.parse(prepare_stub(source, {"Gate"}))
        self.assertEqual(ast.unparse(result.body[0]), "import scaluq")
        self.assertEqual(ast.unparse(result.body[1].returns), "scaluq.Gate")
        self.assertEqual(
            prepare_stub(ast.unparse(result), {"Gate"}), ast.unparse(result) + "\n"
        )

    def test_root_stub_does_not_import_itself(self):
        source = '''
import scaluq.scaluq_core
class GateType: ...
def gate_type() -> scaluq.scaluq_core.GateType: ...
'''
        result = ast.parse(prepare_stub(source))
        self.assertFalse(any(isinstance(node, ast.Import) for node in result.body))
        self.assertEqual(ast.unparse(result.body[-1].returns), "GateType")

    def test_unknown_types_and_import_aliases_are_not_rewritten(self):
        for source in [
            "import scaluq.scaluq_core\ndef f() -> scaluq.scaluq_core.Unknown: ...\n",
            "import scaluq.scaluq_core as backend\ndef f() -> backend.Gate: ...\n",
            "import other.scaluq_core\ndef f() -> other.scaluq_core.Gate: ...\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(
                    ast.dump(ast.parse(prepare_stub(source, {"Gate"}))),
                    ast.dump(ast.parse(source)),
                )

    def test_array_metadata_is_documented_without_changing_call_signature(self):
        source = '''
from typing import Annotated, overload
from numpy.typing import ArrayLike

class State:
    @overload
    def load(self, matrix: Annotated[ArrayLike, dict(shape=(None, None), order='C')], *, seed: int | None = None) -> None:
        """Load the state.

        Args:
            matrix (ArrayLike): Input matrix.

        Examples:
            >>> state.load(matrix)
        """
'''
        tree = ast.parse(prepare_stub(source))
        method = tree.body[2].body[0]
        self.assertEqual(ast.unparse(method.args.args[1].annotation), "ArrayLike")
        self.assertEqual(ast.unparse(method.args.kwonlyargs[0].annotation), "int | None")
        self.assertIsNone(method.args.kw_defaults[0].value)
        self.assertEqual(ast.unparse(method.returns), "None")
        self.assertEqual(ast.unparse(method.decorator_list[0]), "overload")
        doc = ast.get_docstring(method)
        self.assertIn("matrix (ArrayLike): Input matrix.", doc)
        self.assertIn(">>> state.load(matrix)", doc)
        self.assertIn(".. note::", doc)
        self.assertIn("``matrix``: ``dict(shape=(None, None), order='C')``", doc)

    def test_nested_qualified_annotations_and_multiple_metadata(self):
        source = '''
import typing
def matrices() -> list[typing.Annotated[ArrayLike, "two dimensions", dict(order='C')]]:
    ...
'''
        method = ast.parse(prepare_stub(source)).body[1]
        self.assertEqual(ast.unparse(method.returns), "list[ArrayLike]")
        doc = ast.get_docstring(method)
        self.assertIn("``return value``", doc)
        self.assertIn("'two dimensions'", doc)
        self.assertIn("dict(order='C')", doc)
        self.assertIsInstance(method.body[1], ast.Expr)

    def test_unrelated_types_and_example_text_are_preserved(self):
        source = '''
def sample(seed: int | None = None) -> list[int]:
    """Example text: Annotated[ArrayLike, dict(shape=(None, None))]."""
    ...
'''
        self.assertEqual(
            ast.dump(ast.parse(prepare_stub(source))), ast.dump(ast.parse(source))
        )

    def test_preparing_a_stub_twice_does_not_duplicate_notes(self):
        source = '''
def matrix() -> Annotated[ArrayLike, dict(dtype='complex128')]:
    """Return a matrix."""
'''
        once = prepare_stub(source)
        self.assertEqual(prepare_stub(once), once)


if __name__ == "__main__":
    unittest.main()
