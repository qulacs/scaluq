"""Prepare documentation copies of stubs without changing the installed types."""

import ast


class _AnnotationForDocs(ast.NodeTransformer):
    def __init__(self):
        self.metadata = []

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if ast.unparse(node.value) not in {"Annotated", "typing.Annotated"}:
            return node
        if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) < 2:
            return node
        self.metadata.extend(ast.unparse(item) for item in node.slice.elts[1:])
        return ast.copy_location(node.slice.elts[0], node)


def prepare_stub(source):
    """Move Annotated metadata from type annotations into literal documentation.

    Sphinx 7 cannot parse nanobind's dict(...) metadata as a Python type and
    attempts to link dtype, shape and order as class names. Keep the underlying
    type linkable and preserve the metadata in a note on the same function.
    Only the copied stubs consumed by AutoAPI pass through this transformation.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        notes = []
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        annotations = [(arg, "annotation", arg.arg) for arg in args]
        annotations.append((node, "returns", "return value"))
        for owner, attribute, label in annotations:
            annotation = getattr(owner, attribute)
            if annotation is None:
                continue
            formatter = _AnnotationForDocs()
            setattr(owner, attribute, formatter.visit(annotation))
            for metadata in formatter.metadata:
                notes.append(f"Type metadata for ``{label}``: ``{metadata}``.")
        if not notes:
            continue
        original = ast.get_docstring(node)
        note = ".. note::\n\n" + "\n\n".join(f"    {line}" for line in notes)
        docstring = ast.Expr(ast.Constant(f"{original}\n\n{note}" if original else note))
        if original is not None:
            node.body[0] = docstring
        else:
            node.body.insert(0, docstring)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
