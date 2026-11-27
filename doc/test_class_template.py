from pathlib import Path
from types import SimpleNamespace
import unittest

from jinja2 import Environment, FileSystemLoader


class ClassTemplateTest(unittest.TestCase):
    def render(self, typehints, args, overloads, constructor):
        obj = SimpleNamespace(
            display=True,
            type="class",
            short_name="Gate",
            args=args,
            overloads=overloads,
            constructor=constructor,
            app=SimpleNamespace(config=SimpleNamespace(autodoc_typehints=typehints)),
            children=[],
            bases=[],
            docstring="General class of QuantumGate.",
        )
        env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "source/_templates"),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.filters["link_objs"] = str
        return env.get_template("python/class.rst").render(
            obj=obj, is_own_page=False, own_page_types=[], autoapi_options=[]
        )

    def test_description_mode_retains_each_constructor_type(self):
        result = self.render(
            "description",
            "arg: SparseMatrixGate, /",
            [("arg, /", "None"), ("arg, /", "None")],
            SimpleNamespace(overloads=[
                ("self, arg: DenseMatrixGate, /", "None"),
                ("self, arg: XGate, /", "None"),
            ]),
        )
        self.assertIn("Gate(arg: SparseMatrixGate, /)", result)
        self.assertIn("Gate(arg: DenseMatrixGate, /)", result)
        self.assertIn("Gate(arg: XGate, /)", result)
        self.assertNotIn("Gate(arg, /)", result)
        self.assertNotIn("self,", result)
        self.assertIn("General class of QuantumGate.", result)

    def test_none_mode_still_hides_constructor_types(self):
        result = self.render(
            "none", "arg, /", [("arg, /", "None")],
            SimpleNamespace(overloads=[("self, arg: XGate, /", "None")]),
        )
        self.assertIn("Gate(arg, /)", result)
        self.assertNotIn("XGate", result)

    def test_class_without_constructor_keeps_its_description(self):
        result = self.render("description", "", [], None)
        self.assertIn(".. py:class:: Gate", result)
        self.assertIn("General class of QuantumGate.", result)


if __name__ == "__main__":
    unittest.main()
