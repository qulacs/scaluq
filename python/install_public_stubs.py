import ast
import sys
from pathlib import Path


def _node_start(node):
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(decorator.lineno for decorator in decorators)]) - 1


def _replace_node(lines, node, replacement):
    start = _node_start(node)
    end = node.end_lineno
    lines[start:end] = [line if line.endswith("\n") else f"{line}\n" for line in replacement]


def _replace_function_header(lines, node, signature):
    start = node.lineno - 1
    end = node.body[0].lineno - 1
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    lines[start:end] = [f"{indent}{signature}\n"]


def _function_header_replacement(lines, node, signature):
    start = node.lineno - 1
    end = node.body[0].lineno - 1
    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    return start, end, [f"{indent}{signature}\n"]


def _apply_replacements(lines, replacements):
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        lines[start:end] = [
            line if line.endswith("\n") else f"{line}\n" for line in replacement
        ]


def _class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"class {name!r} not found")


def _methods(cls, name):
    return [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _method(cls, name):
    methods = _methods(cls, name)
    if not methods:
        raise RuntimeError(f"method {cls.name}.{name} not found")
    return methods[0]


def _functions(tree, name):
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _function(tree, name):
    functions = _functions(tree, name)
    if not functions:
        raise RuntimeError(f"function {name!r} not found")
    return functions[0]


def _replace_methods(lines, cls, name, replacement):
    methods = _methods(cls, name)
    if not methods:
        return False
    start = _node_start(methods[0])
    end = methods[-1].end_lineno
    lines[start:end] = [
        line if line.endswith("\n") else f"{line}\n" for line in replacement
    ]
    return True


def _methods_replacement(cls, name, replacement):
    methods = _methods(cls, name)
    if not methods:
        return None
    return _node_start(methods[0]), methods[-1].end_lineno, replacement


def _signature_with_extra_keywords(header, extra_keywords):
    stripped = header.strip()
    if not stripped.endswith(":"):
        raise RuntimeError(f"expected function signature ending with ':': {stripped}")
    stripped = stripped[:-1]
    open_paren = stripped.find("(")
    close_paren = stripped.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        raise RuntimeError(f"could not parse function signature: {stripped}")
    params = stripped[open_paren + 1 : close_paren].strip()
    if any(keyword.split(":", 1)[0].split("=", 1)[0].strip() in params for keyword in extra_keywords):
        return f"{stripped}:"
    extra = ", ".join(extra_keywords)
    new_params = f"{params}, {extra}" if params else extra
    return f"{stripped[:open_paren + 1]}{new_params}{stripped[close_paren:]}:"


def _patch_gate_stub(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        start = node.lineno - 1
        end = node.body[0].lineno - 1
        header = "".join(lines[start:end])
        extra = ["precision: str = 'f64'"]
        if node.name in {"DenseMatrix", "SparseMatrix"}:
            extra.append("space: str = 'default'")
        signature = _signature_with_extra_keywords(header, extra)
        _replace_function_header(lines, node, signature)
    return "".join(lines)


def _patch_main_stub(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []

    state_vector = _class(tree, "StateVector")
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector, "__init__"),
        "def __init__(self, n_qubits: int, precision: str = 'f64', space: str = 'default') -> None:",
    ))
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector, "Haar_random_state"),
        "def Haar_random_state(n_qubits: int, seed: int | None = None, precision: str = 'f64', space: str = 'default') -> StateVector:",
    ))
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector, "uninitialized_state"),
        "def uninitialized_state(n_qubits: int, precision: str = 'f64', space: str = 'default') -> StateVector:",
    ))
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector, "inner_product"),
        "def inner_product(a: StateVector, b: StateVector, precision: str = 'f64', space: str = 'default') -> complex:",
    ))

    state_vector_batched = _class(tree, "StateVectorBatched")
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector_batched, "__init__"),
        "def __init__(self, batch_size: int, n_qubits: int, precision: str = 'f64', space: str = 'default') -> None:",
    ))
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector_batched, "Haar_random_state"),
        "def Haar_random_state(batch_size: int, n_qubits: int, set_same_state: bool, seed: int | None = None, precision: str = 'f64', space: str = 'default') -> StateVectorBatched:",
    ))
    replacements.append(_function_header_replacement(
        lines,
        _method(state_vector_batched, "uninitialized_state"),
        "def uninitialized_state(batch_size: int, n_qubits: int, precision: str = 'f64', space: str = 'default') -> StateVectorBatched:",
    ))

    pauli_operator = _class(tree, "PauliOperator")
    init_methods = _methods(pauli_operator, "__init__")
    if init_methods:
        start = _node_start(init_methods[0])
        end = init_methods[-1].end_lineno
        replacements.append((
            start,
            end,
            [
                "    def __init__(self, pauli_string: str = '', coef: complex = 1.0, precision: str = 'f64') -> None:\n",
                '        """Initialize a Pauli operator using pauli_string and the requested precision. See `from_pauli_string` for more details."""\n',
                "\n",
                "    @staticmethod\n",
                "    def from_targets_and_pauli_ids(target_qubit_list: Sequence[int], pauli_id_list: Sequence[int], coef: complex = 1.0, precision: str = 'f64') -> PauliOperator:\n",
                '        """Initialize pauli operator. For each `i`, single pauli correspond to `pauli_id_list[i]` is applied to `target_qubit_list[i]`-th qubit."""\n',
                "\n",
                "    @staticmethod\n",
                "    def from_pauli_string(pauli_string: str, coef: complex = 1.0, precision: str = 'f64') -> PauliOperator:\n",
                '        """Initialize pauli operator. For each `i`, single pauli correspond to `pauli_id_per_qubit[i]` is applied to `i`-th qubit."""\n',
                "\n",
                "    @staticmethod\n",
                "    def from_pauli_id_par_qubit(pauli_id_par_qubit: Sequence[int], coef: complex = 1.0, precision: str = 'f64') -> PauliOperator:\n",
                '        """Initialize pauli operator. For each `i`, single pauli correspond to `pauli_id_per_qubit[i]` is applied to `i`-th qubit."""\n',
                "\n",
                "    @staticmethod\n",
                "    def from_XZ_mask(bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0, precision: str = 'f64') -> PauliOperator:\n",
                '        """\n',
                '        Initialize pauli operator. For each `i`, single pauli applied to `i`-th qubit is got from `i-th` bit of `bit_flip_mask` and `phase_flip_mask` as follows.\n',
                "\n",
                '        .. csv-table::\n',
                "\n",
                '            "bit_flip","phase_flip","pauli"\n',
                '            "0","0","I"\n',
                '            "0","1","Z"\n',
                '            "1","0","X"\n',
                '            "1","1","Y"\n',
                '        """\n'
            ],
        ))

    merge_gate = _function(tree, "merge_gate")
    replacements.append(_function_header_replacement(
        lines,
        merge_gate,
        "def merge_gate(gate1, gate2, prec: str = 'f64', space: str = 'default') -> tuple[scaluq.scaluq_core.default.f64.Gate, float]:",
    ))

    operator = _class(tree, "Operator")
    replacement = _methods_replacement(
        operator,
        "__init__",
        [
            "    def __init__(self, terms, precision: str = 'f64', space: str = 'default') -> None:",
            '        """Initialize an operator from terms using the requested precision and execution space."""',
        ],
    )
    if replacement:
        replacements.append(replacement)

    operator_batched = _class(tree, "OperatorBatched")
    replacement = _methods_replacement(
        operator_batched,
        "__init__",
        [
            "    def __init__(self, terms, precision: str = 'f64', space: str = 'default') -> None:",
            '        """Initialize a batched operator using the requested precision and execution space."""',
        ],
    )
    if replacement:
        replacements.append(replacement)

    circuit = _class(tree, "Circuit")
    replacement = _methods_replacement(
        circuit,
        "__init__",
        [
            "    def __init__(self, precision: str = 'f64') -> None:",
            '        """Initialize an empty circuit using the requested precision."""',
        ],
    )
    if replacement:
        replacements.append(replacement)

    _apply_replacements(lines, replacements)
    return "".join(lines)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    root_stub = Path(argv[1])
    main_stub = Path(argv[2])
    gate_stub = Path(argv[3])
    install_path = Path(argv[4])

    public_root = root_stub.read_text() + "\n" + _patch_main_stub(main_stub.read_text())
    public_gate = _patch_gate_stub(gate_stub.read_text())

    scaluq_path = install_path / "scaluq"
    gate_path = scaluq_path / "gate"
    gate_path.mkdir(parents=True, exist_ok=True)
    (scaluq_path / "__init__.pyi").write_text(public_root)
    (scaluq_path / "py.typed").write_text("")
    (gate_path / "__init__.pyi").write_text(public_gate)
    (gate_path / "py.typed").write_text("")


if __name__ == "__main__":
    main()
