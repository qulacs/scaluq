import ast
import doctest
from pathlib import Path

import scaluq


def add_to_globs(globs, module):
    for name in dir(module):
        if not name.startswith("_"):
            globs[name] = getattr(module, name)


def collect_stub_docstrings(path, module_name):
    tests = []

    def walk(node, qualname):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            current_name = f"{qualname}.{node.name}" if qualname else node.name
            doc = ast.get_docstring(node)
            if doc and ">>> " in doc:
                tests.append((current_name, doc, node.lineno))
            for child in node.body:
                walk(child, current_name)

    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        walk(node, module_name)
    return tests


def installed_stub_paths():
    package_dir = Path(scaluq.__path__[0])
    candidates = [
        (package_dir / "__init__.pyi", "scaluq"),
        (package_dir / "gate" / "__init__.pyi", "scaluq.gate"),
    ]
    return [(path, module_name) for path, module_name in candidates if path.exists()]


globs = globals().copy()
add_to_globs(globs, scaluq)
add_to_globs(globs, scaluq.gate)

parser = doctest.DocTestParser()
runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)

for stub_path, module_name in installed_stub_paths():
    for name, doc, lineno in collect_stub_docstrings(stub_path, module_name):
        print(f"Running doctests for {name}...")
        test = parser.get_doctest(doc, globs.copy(), name, str(stub_path), lineno)
        runner.run(test)

runner.summarize(verbose=True)
