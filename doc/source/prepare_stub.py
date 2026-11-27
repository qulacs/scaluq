"""Prepare documentation copies of stubs without changing the installed types."""

import ast
import copy
import re


class _PublicTypesForDocs(ast.NodeTransformer):
    _backend_type = re.compile(
        r"(?:scaluq\.)?scaluq_core\."
        r"(?:(?:default|host|host_serial)\.(?:f16|f32|f64|bf16)\.)?"
        r"(?P<name>[A-Za-z_]\w*)"
    )

    def __init__(self, public_types, local_types):
        self.public_types = public_types
        self.local_types = local_types
        self.needs_public_import = False

    def visit_Attribute(self, node):
        match = self._backend_type.fullmatch(ast.unparse(node))
        if match and match["name"] in self.public_types:
            name = match["name"]
            if name in self.local_types:
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
            self.needs_public_import = True
            return ast.copy_location(ast.parse(f"scaluq.{name}", mode="eval").body, node)
        return self.generic_visit(node)


def _public_types_for_docs(tree, public_types):
    local_types = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    formatter = _PublicTypesForDocs(set(public_types) | local_types, local_types)
    tree = formatter.visit(tree)
    # Retain backend imports if an unknown type still needs them. Never invent
    # a public type just to resolve a documentation link.
    needs_backend = any(
        isinstance(node, (ast.Attribute, ast.Name))
        and (
            ast.unparse(node) in {"scaluq.scaluq_core", "scaluq_core"}
            or ast.unparse(node).startswith(("scaluq.scaluq_core.", "scaluq_core."))
        )
        for node in ast.walk(tree)
    )
    if not needs_backend:
        for node in tree.body[:]:
            if isinstance(node, ast.Import):
                node.names = [
                    alias
                    for alias in node.names
                    if alias.asname is not None
                    or not (
                        alias.name in {"scaluq.scaluq_core", "scaluq_core"}
                        or alias.name.startswith(("scaluq.scaluq_core.", "scaluq_core."))
                    )
                ]
                if not node.names:
                    tree.body.remove(node)
    if formatter.needs_public_import and not any(
        isinstance(node, ast.Import)
        and any(alias.name == "scaluq" and alias.asname is None for alias in node.names)
        for node in tree.body
    ):
        tree.body.insert(0, ast.Import(names=[ast.alias(name="scaluq")]))
    return tree


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


def _signature_for_docs(node, is_method):
    args = copy.deepcopy(node.args)
    positional = args.posonlyargs or args.args
    if is_method and positional and positional[0].arg in {"self", "cls"}:
        positional.pop(0)
    signature = f"{node.name}({ast.unparse(args)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _merge_overload_docs(tree):
    """Keep all calling forms and distinct documentation under one API item.

    AutoAPI groups overloads but keeps only the last definition's docstring.
    Give every definition the merged text so no explanation is overwritten.
    This also makes preparing a documentation stub a second time idempotent.
    """
    for owner in ast.walk(tree):
        if not isinstance(owner, (ast.Module, ast.ClassDef)):
            continue
        groups = {}
        for node in owner.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                groups.setdefault(node.name, []).append(node)
        removed = set()
        for group in groups.values():
            if len(group) < 2 or not any(
                ast.unparse(decorator) in {"overload", "typing.overload"}
                for node in group for decorator in node.decorator_list
            ):
                continue
            docs = {}
            signatures = set()
            for node in group:
                signature = _signature_for_docs(node, isinstance(owner, ast.ClassDef))
                doc = ast.get_docstring(node)
                if doc:
                    forms = docs.setdefault(doc, [])
                    if signature not in forms:
                        forms.append(signature)
                key = (
                    ast.dump(node.args),
                    ast.dump(node.returns) if node.returns is not None else None,
                    tuple(ast.dump(decorator) for decorator in node.decorator_list),
                )
                if key in signatures:
                    removed.add(id(node))
                signatures.add(key)
            if len(docs) == 1:
                merged = next(iter(docs))
            elif docs:
                common = next(iter(docs)).split("\n\n", 1)[0]
                if not all(doc == common or doc.startswith(common + "\n\n") for doc in docs):
                    common = ""
                sections = [common] if common else []
                for doc, forms in docs.items():
                    detail = doc[len(common):].lstrip("\n") if common else doc
                    if detail:
                        sections.append(
                            ".. rubric:: Calling forms\n\n.. code-block:: python\n\n"
                            + "\n".join(f"    {form}" for form in forms)
                            + "\n\n" + detail
                        )
                merged = "\n\n".join(sections)
            else:
                continue
            for node in group:
                docstring = ast.Expr(ast.Constant(merged))
                if ast.get_docstring(node) is not None:
                    node.body[0] = docstring
                else:
                    node.body.insert(0, docstring)
        owner.body = [node for node in owner.body if id(node) not in removed]
    return tree


def prepare_stub(source, public_types=()):
    """Move Annotated metadata from type annotations into literal documentation.

    Sphinx 7 cannot parse nanobind's dict(...) metadata as a Python type and
    attempts to link dtype, shape and order as class names. Keep the underlying
    type linkable and preserve the metadata in a note on the same function.
    Only the copied stubs consumed by AutoAPI pass through this transformation.
    Known backend types link to the public classes declared in the root stub;
    installed annotations retain their precision and execution-space details.
    """
    tree = _public_types_for_docs(ast.parse(source), public_types)
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
    return ast.unparse(ast.fix_missing_locations(_merge_overload_docs(tree))) + "\n"
