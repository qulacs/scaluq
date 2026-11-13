import ast
import unittest

from source.prepare_stub import prepare_stub


class PrepareStubTest(unittest.TestCase):
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
