#include <scaluq/prec_space.hpp>
#include <scaluq/qasm/qasm2.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

namespace scaluq::qasm2 {
namespace {

struct Location {
    std::uint64_t line = 1;
    std::uint64_t column = 1;
};

struct Statement {
    std::string text;
    Location loc;
};

struct Register {
    std::uint64_t offset;
    std::uint64_t size;
};

struct RegisterOperand {
    std::uint64_t offset;
    std::uint64_t size;
};

// Result type produced by ExprParser.  It represents the numeric subset of
// OpenQASM angle expressions that Scaluq can lower.
struct LinearExpr {
    double constant = 0.;
};

// Export-side operation records used to describe what a Scaluq gate means
// before choosing an OpenQASM dialect.  The classifier below produces one of
// these records from Gate/ParamGate, and the QASM2 lowering step consumes it.
struct IgnoredOp {};

struct NamedUnitaryOp {
    std::string_view name;
    std::vector<std::uint64_t> targets;
    std::vector<std::uint64_t> controls;
    std::vector<std::uint64_t> control_values;
};

struct RotationOp {
    std::string_view name;
    double angle;
    std::uint64_t target;
    std::vector<std::uint64_t> controls;
    std::vector<std::uint64_t> control_values;
};

struct UGateOp {
    std::string_view name;
    std::vector<double> params;
    std::uint64_t target;
    std::vector<std::uint64_t> controls;
    std::vector<std::uint64_t> control_values;
};

struct MeasurementOp {
    std::uint64_t target;
    std::uint64_t classical_bit;
    bool reset;
};

struct UnsupportedCircuitOp {
    std::string reason;
};

using CircuitOpView = std::variant<IgnoredOp,
                                   NamedUnitaryOp,
                                   RotationOp,
                                   UGateOp,
                                   MeasurementOp,
                                   UnsupportedCircuitOp>;

// Concrete OpenQASM 2.0 operation used by the writer.  At this point qelib1
// names, parameter text, qubit order, and optional classical-bit destination
// have already been decided by the lowering layer.
struct OpenQasm2Op {
    enum class Kind { Gate, Measure };

    Kind kind;
    std::string name;
    std::vector<std::string> params;
    std::vector<std::uint64_t> qubits;
    std::optional<std::uint64_t> classical_bit;
};

// Build a location-aware parser error message for a statement-level parser
// helper.  Callers pass the Location where the offending statement started.
[[nodiscard]] std::string make_error(const Location& loc, std::string_view message) {
    std::ostringstream oss;
    oss << "OpenQASM 2.0 error at line " << loc.line << ", column " << loc.column << ": "
        << message;
    return oss.str();
}

[[nodiscard]] std::string trim(std::string_view s) {
    auto is_space = [](unsigned char c) { return std::isspace(c); };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return std::string(s);
}

[[nodiscard]] std::string lower(std::string_view s) {
    std::string out(s);
    std::ranges::transform(out, out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

[[nodiscard]] bool starts_with_word(std::string_view s, std::string_view word) {
    if (!s.starts_with(word)) return false;
    if (s.size() == word.size()) return true;
    unsigned char c = static_cast<unsigned char>(s[word.size()]);
    return !(std::isalnum(c) || c == '_');
}

// Split source text into semicolon-terminated OpenQASM statements while
// ignoring comments and preserving the start location of each statement.
[[nodiscard]] std::vector<Statement> split_statements(std::string_view source) {
    std::vector<Statement> statements;
    std::string current;
    Location current_loc{1, 1};
    Location loc{1, 1};
    bool has_content = false;
    bool in_line_comment = false;
    bool in_block_comment = false;
    bool in_string = false;

    auto advance = [&loc](char c) {
        if (c == '\n') {
            ++loc.line;
            loc.column = 1;
        } else {
            ++loc.column;
        }
    };

    for (std::size_t i = 0; i < source.size(); ++i) {
        char c = source[i];
        char next = i + 1 < source.size() ? source[i + 1] : '\0';

        if (in_line_comment) {
            if (c == '\n') in_line_comment = false;
            advance(c);
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && next == '/') {
                advance(c);
                advance(next);
                ++i;
                in_block_comment = false;
                continue;
            }
            advance(c);
            continue;
        }
        if (!in_string && c == '/' && next == '/') {
            in_line_comment = true;
            advance(c);
            advance(next);
            ++i;
            continue;
        }
        if (!in_string && c == '/' && next == '*') {
            in_block_comment = true;
            advance(c);
            advance(next);
            ++i;
            continue;
        }

        if (!has_content && !std::isspace(static_cast<unsigned char>(c))) {
            current_loc = loc;
            has_content = true;
        }
        if (c == '"') in_string = !in_string;
        if (!in_string && c == ';') {
            std::string stmt = trim(current);
            if (!stmt.empty()) statements.push_back({stmt, current_loc});
            current.clear();
            has_content = false;
        } else {
            current.push_back(c);
        }
        advance(c);
    }
    if (in_block_comment) {
        throw std::runtime_error(make_error(loc, "unterminated block comment"));
    }
    if (in_string) {
        throw std::runtime_error(make_error(loc, "unterminated string literal"));
    }
    if (!trim(current).empty()) {
        throw std::runtime_error(make_error(current_loc, "statement is missing ';'"));
    }
    return statements;
}

// Parser for OpenQASM angle expressions.  It converts numeric expressions such
// as "pi / 2" into a LinearExpr used to create ordinary rotation gates.
class ExprParser {
public:
    ExprParser(std::string_view source, Location loc) : _source(source), _loc(loc) {}

    LinearExpr parse() {
        LinearExpr result = parse_add_sub();
        skip_spaces();
        if (_pos != _source.size()) {
            throw std::runtime_error(error("unexpected token in angle expression"));
        }
        return result;
    }

private:
    std::string_view _source;
    Location _loc;
    std::size_t _pos = 0;

    [[nodiscard]] std::string error(std::string_view message) const {
        return make_error(_loc, message);
    }

    void skip_spaces() {
        while (_pos < _source.size() &&
               std::isspace(static_cast<unsigned char>(_source[_pos]))) {
            ++_pos;
        }
    }

    [[nodiscard]] bool consume(char c) {
        skip_spaces();
        if (_pos < _source.size() && _source[_pos] == c) {
            ++_pos;
            return true;
        }
        return false;
    }

    [[nodiscard]] LinearExpr parse_add_sub() {
        LinearExpr lhs = parse_mul_div();
        while (true) {
            if (consume('+')) {
                lhs = add(lhs, parse_mul_div());
            } else if (consume('-')) {
                lhs = add(lhs, negate(parse_mul_div()));
            } else {
                return lhs;
            }
        }
    }

    [[nodiscard]] LinearExpr parse_mul_div() {
        LinearExpr lhs = parse_unary();
        while (true) {
            if (consume('*')) {
                lhs = multiply(lhs, parse_unary());
            } else if (consume('/')) {
                LinearExpr rhs = parse_unary();
                if (rhs.constant == 0.) {
                    throw std::runtime_error(error("division by zero"));
                }
                lhs.constant /= rhs.constant;
            } else {
                return lhs;
            }
        }
    }

    [[nodiscard]] LinearExpr parse_unary() {
        if (consume('+')) return parse_unary();
        if (consume('-')) return negate(parse_unary());
        return parse_primary();
    }

    [[nodiscard]] LinearExpr parse_primary() {
        skip_spaces();
        if (consume('(')) {
            LinearExpr result = parse_add_sub();
            if (!consume(')')) throw std::runtime_error(error("expected ')'"));
            return result;
        }
        if (_pos >= _source.size()) throw std::runtime_error(error("expected expression"));

        unsigned char c = static_cast<unsigned char>(_source[_pos]);
        if (std::isalpha(c) || c == '_') {
            std::size_t begin = _pos++;
            while (_pos < _source.size()) {
                unsigned char idc = static_cast<unsigned char>(_source[_pos]);
                if (!(std::isalnum(idc) || idc == '_')) break;
                ++_pos;
            }
            std::string id(_source.substr(begin, _pos - begin));
            if (id == "pi") {
                return {.constant = std::numbers::pi};
            }
            throw std::runtime_error(error("only the OpenQASM constant 'pi' is supported in angle expressions"));
        }

        char* end = nullptr;
        std::string rest(_source.substr(_pos));
        double value = std::strtod(rest.c_str(), &end);
        if (end == rest.c_str()) throw std::runtime_error(error("expected number or identifier"));
        _pos += static_cast<std::size_t>(end - rest.c_str());
        return {.constant = value};
    }

    [[nodiscard]] LinearExpr add(LinearExpr lhs, const LinearExpr& rhs) const {
        lhs.constant += rhs.constant;
        return lhs;
    }

    [[nodiscard]] LinearExpr negate(LinearExpr expr) const {
        expr.constant = -expr.constant;
        return expr;
    }

    [[nodiscard]] LinearExpr multiply(const LinearExpr& lhs, const LinearExpr& rhs) const {
        return {.constant = lhs.constant * rhs.constant};
    }
};

[[nodiscard]] std::vector<std::string> split_comma_list(std::string_view source, Location loc) {
    std::vector<std::string> parts;
    std::size_t begin = 0;
    int depth = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (source[i] == '(') ++depth;
        if (source[i] == ')') --depth;
        if (depth < 0) throw std::runtime_error(make_error(loc, "unmatched ')'"));
        if (source[i] == ',' && depth == 0) {
            parts.push_back(trim(source.substr(begin, i - begin)));
            begin = i + 1;
        }
    }
    if (depth != 0) throw std::runtime_error(make_error(loc, "unmatched '('"));
    if (begin < source.size()) parts.push_back(trim(source.substr(begin)));
    if (parts.size() == 1 && parts[0].empty()) parts.clear();
    return parts;
}

// Parse an indexed OpenQASM register operand and return the flattened Scaluq
// qubit/classical-bit index.  The register map supplies each declared
// register's offset in the single flat index space used by Circuit.
template <typename Registers>
[[nodiscard]] std::uint64_t parse_indexed_register(std::string_view operand,
                                                   const Registers& registers,
                                                   Location loc,
                                                   std::string_view kind) {
    std::string op = trim(operand);
    std::size_t lb = op.find('[');
    std::size_t rb = op.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb + 1 != op.size() || rb < lb) {
        throw std::runtime_error(make_error(loc, "expected indexed register operand"));
    }
    std::string name = trim(std::string_view(op).substr(0, lb));
    auto it = registers.find(name);
    if (it == registers.end()) {
        throw std::runtime_error(make_error(
            loc, std::string("unknown ") + std::string(kind) + " register '" + name + "'"));
    }
    std::string index_text = trim(std::string_view(op).substr(lb + 1, rb - lb - 1));
    char* end = nullptr;
    unsigned long long index = std::strtoull(index_text.c_str(), &end, 10);
    if (end == index_text.c_str() || *end != '\0') {
        throw std::runtime_error(make_error(loc, "register index must be an integer"));
    }
    if (index >= it->second.size) {
        throw std::runtime_error(make_error(loc, "register index out of range"));
    }
    return it->second.offset + index;
}

// Parse a measurement operand, which OpenQASM permits to be either a single
// indexed bit/qubit or a whole register.
template <typename Registers>
[[nodiscard]] RegisterOperand parse_measure_register_operand(std::string_view operand,
                                                             const Registers& registers,
                                                             Location loc,
                                                             std::string_view kind) {
    std::string op = trim(operand);
    if (op.find('[') != std::string::npos || op.find(']') != std::string::npos) {
        return RegisterOperand{parse_indexed_register(op, registers, loc, kind), 1};
    }
    auto it = registers.find(op);
    if (it == registers.end()) {
        throw std::runtime_error(make_error(
            loc, std::string("unknown ") + std::string(kind) + " register '" + op + "'"));
    }
    return RegisterOperand{it->second.offset, it->second.size};
}

[[nodiscard]] std::uint64_t parse_register_decl(std::string_view stmt,
                                                std::string_view keyword,
                                                Location loc) {
    std::string s = trim(stmt.substr(keyword.size()));
    std::size_t lb = s.find('[');
    std::size_t rb = s.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb + 1 != s.size() || rb < lb) {
        throw std::runtime_error(make_error(loc, "invalid register declaration"));
    }
    std::string size_text = trim(std::string_view(s).substr(lb + 1, rb - lb - 1));
    char* end = nullptr;
    unsigned long long size = std::strtoull(size_text.c_str(), &end, 10);
    if (end == size_text.c_str() || *end != '\0' || size == 0) {
        throw std::runtime_error(make_error(loc, "register size must be a positive integer"));
    }
    return size;
}

[[nodiscard]] std::string parse_register_name(std::string_view stmt,
                                              std::string_view keyword,
                                              Location loc) {
    std::string s = trim(stmt.substr(keyword.size()));
    std::size_t lb = s.find('[');
    if (lb == std::string::npos) throw std::runtime_error(make_error(loc, "invalid register declaration"));
    return trim(std::string_view(s).substr(0, lb));
}

[[nodiscard]] bool all_control_values_are_one(const std::vector<std::uint64_t>& control_values) {
    for (std::uint64_t value : control_values) {
        if (value != 1) return false;
    }
    return true;
}

[[nodiscard]] std::string format_angle(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

template <Precision Prec>
void add_rotation(Circuit<Prec>& circuit,
                  std::string_view name,
                  std::uint64_t target,
                  const LinearExpr& angle,
                  const std::vector<std::uint64_t>& controls,
                  Location loc) {
    if (name == "rx")
        circuit.add_gate(gate::RX<Prec>(target, angle.constant, controls));
    else if (name == "ry")
        circuit.add_gate(gate::RY<Prec>(target, angle.constant, controls));
    else if (name == "rz")
        circuit.add_gate(gate::RZ<Prec>(target, angle.constant, controls));
    else
        throw std::runtime_error(make_error(loc, "unsupported rotation gate"));
}

// Reads OpenQASM 2.0 statements and builds the public Qasm2Circuit load result.
// It owns the register maps used to translate QASM register operands into
// Scaluq's flat qubit/classical-bit indices.
template <Precision Prec>
class Reader {
public:
    Qasm2Circuit<Prec> read(std::string_view source) {
        for (const Statement& stmt : split_statements(source)) {
            handle_statement(stmt);
        }
        return std::move(_result);
    }

private:
    Qasm2Circuit<Prec> _result;
    std::map<std::string, Register> _qregs;
    std::map<std::string, Register> _cregs;
    bool _saw_version = false;

    // Dispatch one OpenQASM statement to the appropriate reader routine and
    // update either the output Circuit or the QASM-only metadata.
    void handle_statement(const Statement& statement) {
        std::string stmt = trim(statement.text);
        std::string low = lower(stmt);
        if (starts_with_word(low, "openqasm") &&
            trim(std::string_view(low).substr(std::string_view("openqasm").size())) == "2.0") {
            _saw_version = true;
            return;
        }
        if (starts_with_word(low, "include")) {
            if (trim(std::string_view(low).substr(std::string_view("include").size())) !=
                "\"qelib1.inc\"") {
                throw std::runtime_error(make_error(statement.loc, "only include \"qelib1.inc\" is supported"));
            }
            return;
        }
        if (starts_with_word(low, "qreg")) {
            std::string name = parse_register_name(stmt, "qreg", statement.loc);
            if (_qregs.contains(name)) throw std::runtime_error(make_error(statement.loc, "duplicate qreg"));
            std::uint64_t size = parse_register_decl(stmt, "qreg", statement.loc);
            _qregs.emplace(name, Register{_result.n_qubits, size});
            _result.n_qubits += size;
            return;
        }
        if (starts_with_word(low, "creg")) {
            std::string name = parse_register_name(stmt, "creg", statement.loc);
            if (_cregs.contains(name)) throw std::runtime_error(make_error(statement.loc, "duplicate creg"));
            std::uint64_t size = parse_register_decl(stmt, "creg", statement.loc);
            _cregs.emplace(name, Register{_result.n_clbits, size});
            _result.n_clbits += size;
            return;
        }
        if (starts_with_word(low, "barrier")) {
            _result.warnings.push_back("barrier is ignored");
            return;
        }
        if (starts_with_word(low, "measure")) {
            parse_measure(stmt, statement.loc);
            return;
        }
        for (std::string_view unsupported : {"reset", "if", "gate", "opaque"}) {
            if (starts_with_word(low, unsupported)) {
                throw std::runtime_error(
                    make_error(statement.loc, std::string(unsupported) + " is not supported yet"));
            }
        }
        parse_gate(stmt, statement.loc);
    }

    // Parse a supported qelib-style gate invocation and append the matching
    // Scaluq gate to the output Circuit.
    void parse_gate(std::string_view stmt, Location loc) {
        if (!_saw_version) {
            _result.warnings.push_back("OPENQASM 2.0 header is missing");
            _saw_version = true;
        }
        std::size_t name_end = 0;
        while (name_end < stmt.size()) {
            unsigned char c = static_cast<unsigned char>(stmt[name_end]);
            if (!(std::isalnum(c) || c == '_')) break;
            ++name_end;
        }
        if (name_end == 0) throw std::runtime_error(make_error(loc, "expected gate name"));
        std::string name = lower(stmt.substr(0, name_end));
        if (name == "u") name = "u3";

        std::string rest = trim(stmt.substr(name_end));
        std::vector<LinearExpr> params;
        if (!rest.empty() && rest.front() == '(') {
            int depth = 0;
            std::size_t close = std::string::npos;
            for (std::size_t i = 0; i < rest.size(); ++i) {
                if (rest[i] == '(') ++depth;
                if (rest[i] == ')') --depth;
                if (depth == 0) {
                    close = i;
                    break;
                }
            }
            if (close == std::string::npos) throw std::runtime_error(make_error(loc, "unmatched '('"));
            for (const std::string& param : split_comma_list(std::string_view(rest).substr(1, close - 1), loc)) {
                params.push_back(ExprParser(param, loc).parse());
            }
            rest = trim(std::string_view(rest).substr(close + 1));
        }
        std::vector<std::uint64_t> qubits;
        for (const std::string& operand : split_comma_list(rest, loc)) {
            qubits.push_back(parse_indexed_register(operand, _qregs, loc, "quantum"));
        }
        std::vector<std::uint64_t> sorted_qubits = qubits;
        std::ranges::sort(sorted_qubits);
        if (std::ranges::adjacent_find(sorted_qubits) != sorted_qubits.end()) {
            throw std::runtime_error(make_error(loc, "gate operands must be distinct qubits"));
        }
        add_gate(name, params, qubits, loc);
    }

    void expect(std::string_view name,
                const std::vector<LinearExpr>& params,
                const std::vector<std::uint64_t>& qubits,
                std::size_t n_params,
                std::size_t n_qubits,
                Location loc) {
        if (params.size() != n_params || qubits.size() != n_qubits) {
            std::ostringstream oss;
            oss << name << " expects " << n_params << " parameter(s) and " << n_qubits
                << " qubit(s)";
            throw std::runtime_error(make_error(loc, oss.str()));
        }
    }

    void add_gate(std::string_view name,
                  const std::vector<LinearExpr>& params,
                  const std::vector<std::uint64_t>& qubits,
                  Location loc) {
        if (name == "id") {
            expect(name, params, qubits, 0, 1, loc);
            return;
        }
        if (name == "x" || name == "y" || name == "z" || name == "h" || name == "s" ||
            name == "sdg" || name == "t" || name == "tdg") {
            expect(name, params, qubits, 0, 1, loc);
            if (name == "x") _result.circuit.add_gate(gate::X<Prec>(qubits[0]));
            if (name == "y") _result.circuit.add_gate(gate::Y<Prec>(qubits[0]));
            if (name == "z") _result.circuit.add_gate(gate::Z<Prec>(qubits[0]));
            if (name == "h") _result.circuit.add_gate(gate::H<Prec>(qubits[0]));
            if (name == "s") _result.circuit.add_gate(gate::S<Prec>(qubits[0]));
            if (name == "sdg") _result.circuit.add_gate(gate::Sdag<Prec>(qubits[0]));
            if (name == "t") _result.circuit.add_gate(gate::T<Prec>(qubits[0]));
            if (name == "tdg") _result.circuit.add_gate(gate::Tdag<Prec>(qubits[0]));
            return;
        }
        if (name == "rx" || name == "ry" || name == "rz") {
            expect(name, params, qubits, 1, 1, loc);
            add_rotation(_result.circuit, name, qubits[0], params[0], {}, loc);
            return;
        }
        if (name == "u1") {
            expect(name, params, qubits, 1, 1, loc);
            _result.circuit.add_gate(gate::U1<Prec>(qubits[0], params[0].constant));
            return;
        }
        if (name == "u2" || name == "u3") {
            expect(name, params, qubits, name == "u2" ? 2 : 3, 1, loc);
            if (name == "u2")
                _result.circuit.add_gate(gate::U2<Prec>(qubits[0], params[0].constant, params[1].constant));
            else
                _result.circuit.add_gate(
                    gate::U3<Prec>(qubits[0], params[0].constant, params[1].constant, params[2].constant));
            return;
        }
        if (name == "cx" || name == "cnot") {
            expect(name, params, qubits, 0, 2, loc);
            _result.circuit.add_gate(gate::CX<Prec>(qubits[0], qubits[1]));
            return;
        }
        if (name == "cy" || name == "cz" || name == "ch") {
            expect(name, params, qubits, 0, 2, loc);
            if (name == "cy") _result.circuit.add_gate(gate::Y<Prec>(qubits[1], {qubits[0]}));
            if (name == "cz") _result.circuit.add_gate(gate::CZ<Prec>(qubits[0], qubits[1]));
            if (name == "ch") _result.circuit.add_gate(gate::H<Prec>(qubits[1], {qubits[0]}));
            return;
        }
        if (name == "crx" || name == "cry" || name == "crz") {
            expect(name, params, qubits, 1, 2, loc);
            add_rotation(_result.circuit, name.substr(1), qubits[1], params[0], {qubits[0]}, loc);
            return;
        }
        if (name == "cu1") {
            expect(name, params, qubits, 1, 2, loc);
            _result.circuit.add_gate(gate::U1<Prec>(qubits[1], params[0].constant, {qubits[0]}));
            return;
        }
        if (name == "cu3") {
            expect(name, params, qubits, 3, 2, loc);
            _result.circuit.add_gate(gate::U3<Prec>(
                qubits[1], params[0].constant, params[1].constant, params[2].constant, {qubits[0]}));
            return;
        }
        if (name == "ccx") {
            expect(name, params, qubits, 0, 3, loc);
            _result.circuit.add_gate(gate::CCX<Prec>(qubits[0], qubits[1], qubits[2]));
            return;
        }
        if (name == "swap") {
            expect(name, params, qubits, 0, 2, loc);
            _result.circuit.add_gate(gate::Swap<Prec>(qubits[0], qubits[1]));
            return;
        }
        if (name == "cswap") {
            expect(name, params, qubits, 0, 3, loc);
            _result.circuit.add_gate(gate::Swap<Prec>(qubits[1], qubits[2], {qubits[0]}));
            return;
        }
        throw std::runtime_error(make_error(loc, "unsupported gate '" + std::string(name) + "'"));
    }

    // Parse an OpenQASM measurement statement and append the matching Scaluq
    // Measurement gate with its destination classical-bit index.
    void parse_measure(std::string_view stmt, Location loc) {
        std::string rest = trim(stmt.substr(std::string_view("measure").size()));
        std::size_t arrow = rest.find("->");
        if (arrow == std::string::npos) {
            throw std::runtime_error(make_error(loc, "measurement expects '->'"));
        }
        std::string qoperand = trim(std::string_view(rest).substr(0, arrow));
        std::string coperand = trim(std::string_view(rest).substr(arrow + 2));
        RegisterOperand qubits = parse_measure_register_operand(qoperand, _qregs, loc, "quantum");
        RegisterOperand clbits = parse_measure_register_operand(coperand, _cregs, loc, "classical");
        if (qubits.size != clbits.size) {
            throw std::runtime_error(make_error(loc, "measurement register sizes must match"));
        }
        for (std::uint64_t i = 0; i < qubits.size; ++i) {
            _result.circuit.add_gate(gate::Measurement<Prec>(qubits.offset + i, clbits.offset + i));
        }
    }
};

// Return the single target qubit required by one-target export records.  This
// catches malformed internal records before they reach the QASM writer.
[[nodiscard]] std::uint64_t single_target(const std::vector<std::uint64_t>& targets,
                                          std::string_view type) {
    if (targets.size() != 1) {
        throw std::runtime_error("unexpected target count for " + std::string(type));
    }
    return targets[0];
}

[[nodiscard]] std::uint64_t required_qubits(const OpenQasm2Op& op) {
    std::uint64_t n_qubits = 0;
    for (std::uint64_t qubit : op.qubits) n_qubits = std::max(n_qubits, qubit + 1);
    return n_qubits;
}

// Compute the classical register size required by a lowered QASM2 operation.
// Non-measurement operations do not require classical bits.
[[nodiscard]] std::uint64_t required_clbits(const OpenQasm2Op& op) {
    if (op.kind != OpenQasm2Op::Kind::Measure) return 0;
    return op.classical_bit.value_or(0) + 1;
}

[[nodiscard]] std::string q(std::uint64_t index) {
    return "q[" + std::to_string(index) + "]";
}

[[nodiscard]] std::string c(std::uint64_t index) {
    return "c[" + std::to_string(index) + "]";
}

// Convert a non-parametric Scaluq Gate into the export operation record that
// represents the gate's meaning independently of any concrete QASM dialect.
template <Precision Prec>
[[nodiscard]] CircuitOpView classify_gate_for_export(const Gate<Prec>& gate) {
    const std::vector<std::uint64_t> targets = gate->target_qubit_list();
    const std::vector<std::uint64_t> controls = gate->control_qubit_list();
    const std::vector<std::uint64_t> control_values = gate->control_value_list();

    switch (gate.gate_type()) {
        case GateType::I:
            return IgnoredOp{};
        case GateType::X:
            return NamedUnitaryOp{"x", targets, controls, control_values};
        case GateType::Y:
            return NamedUnitaryOp{"y", targets, controls, control_values};
        case GateType::Z:
            return NamedUnitaryOp{"z", targets, controls, control_values};
        case GateType::H:
            return NamedUnitaryOp{"h", targets, controls, control_values};
        case GateType::S:
            return NamedUnitaryOp{"s", targets, controls, control_values};
        case GateType::Sdag:
            return NamedUnitaryOp{"sdg", targets, controls, control_values};
        case GateType::T:
            return NamedUnitaryOp{"t", targets, controls, control_values};
        case GateType::Tdag:
            return NamedUnitaryOp{"tdg", targets, controls, control_values};
        case GateType::Swap:
            return NamedUnitaryOp{"swap", targets, controls, control_values};
        case GateType::RX:
            return RotationOp{
                "rx", RXGate<Prec>(gate)->angle(), single_target(targets, "RX"), controls, control_values};
        case GateType::RY:
            return RotationOp{
                "ry", RYGate<Prec>(gate)->angle(), single_target(targets, "RY"), controls, control_values};
        case GateType::RZ:
            return RotationOp{
                "rz", RZGate<Prec>(gate)->angle(), single_target(targets, "RZ"), controls, control_values};
        case GateType::U1:
            return UGateOp{
                "u1", {U1Gate<Prec>(gate)->lambda()}, single_target(targets, "U1"), controls, control_values};
        case GateType::U2:
            return UGateOp{"u2",
                           {U2Gate<Prec>(gate)->phi(), U2Gate<Prec>(gate)->lambda()},
                           single_target(targets, "U2"),
                           controls,
                           control_values};
        case GateType::U3:
            return UGateOp{"u3",
                           {U3Gate<Prec>(gate)->theta(), U3Gate<Prec>(gate)->phi(), U3Gate<Prec>(gate)->lambda()},
                           single_target(targets, "U3"),
                           controls,
                           control_values};
        case GateType::Measurement:
            return MeasurementOp{single_target(targets, "Measurement"),
                                 MeasurementGate<Prec>(gate)->classical_bit_index(),
                                 MeasurementGate<Prec>(gate)->reset()};
        case GateType::Unknown:
        case GateType::GlobalPhase:
        case GateType::SqrtX:
        case GateType::SqrtXdag:
        case GateType::SqrtY:
        case GateType::SqrtYdag:
        case GateType::P0:
        case GateType::P1:
        case GateType::Ecr:
        case GateType::Pauli:
        case GateType::PauliRotation:
        case GateType::SparseMatrix:
        case GateType::DenseMatrix:
        case GateType::Probabilistic:
            return UnsupportedCircuitOp{"unsupported gate for OpenQASM 2.0 export"};
    }
    return UnsupportedCircuitOp{"unsupported gate for OpenQASM 2.0 export"};
}

// Convert a Scaluq ParamGate plus its parameter key into the export operation
// record used by the QASM lowering layer.
template <Precision Prec>
[[nodiscard]] CircuitOpView classify_param_gate_for_export(
    const std::pair<ParamGate<Prec>, std::string>& gate_with_key) {
    switch (gate_with_key.first.param_gate_type()) {
        case ParamGateType::ParamRX:
        case ParamGateType::ParamRY:
        case ParamGateType::ParamRZ:
            return UnsupportedCircuitOp{"OpenQASM 2.0 export does not support parametric gates"};
        case ParamGateType::Unknown:
        case ParamGateType::ParamPauliRotation:
        case ParamGateType::ParamProbabilistic:
        case ParamGateType::Error:
            return UnsupportedCircuitOp{"unsupported param gate for OpenQASM 2.0 export"};
    }
    return UnsupportedCircuitOp{"unsupported param gate for OpenQASM 2.0 export"};
}

// Lower a named unitary export record to a concrete OpenQASM 2.0 operation,
// applying qelib1 names such as "cx", "ccx", "cz", "ch", and "cswap".
[[nodiscard]] OpenQasm2Op lower_named_unitary_to_qasm2(const NamedUnitaryOp& op) {
    if (!all_control_values_are_one(op.control_values)) {
        throw std::runtime_error("OpenQASM 2.0 export only supports control value 1");
    }
    if (op.name == "x" && op.controls.size() == 1) {
        return {OpenQasm2Op::Kind::Gate,
                "cx",
                {},
                {op.controls[0], single_target(op.targets, "X")},
                std::nullopt};
    }
    if (op.name == "z" && op.controls.size() == 1) {
        return {OpenQasm2Op::Kind::Gate,
                "cz",
                {},
                {op.controls[0], single_target(op.targets, "Z")},
                std::nullopt};
    }
    if (op.name == "x" && op.controls.size() == 2) {
        return {OpenQasm2Op::Kind::Gate,
                "ccx",
                {},
                {op.controls[0], op.controls[1], single_target(op.targets, "X")},
                std::nullopt};
    }
    if (op.name == "swap" && op.controls.size() == 1) {
        if (op.targets.size() != 2) throw std::runtime_error("unexpected target count for Swap");
        return {OpenQasm2Op::Kind::Gate, "cswap", {}, {op.controls[0], op.targets[0], op.targets[1]}, std::nullopt};
    }
    if (op.controls.size() > 1) {
        throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
    }
    if (op.controls.size() == 1) {
        if (op.name == "y" || op.name == "h") {
            return {OpenQasm2Op::Kind::Gate,
                    "c" + std::string(op.name),
                    {},
                    {op.controls[0], single_target(op.targets, op.name)},
                    std::nullopt};
        }
        throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
    }
    return {OpenQasm2Op::Kind::Gate, std::string(op.name), {}, op.targets, std::nullopt};
}

// Lower a numeric rotation export record to a concrete OpenQASM 2.0 operation,
// choosing "rx/ry/rz" or "crx/cry/crz" based on the control list.
[[nodiscard]] OpenQasm2Op lower_rotation_to_qasm2(const RotationOp& op) {
    if (!all_control_values_are_one(op.control_values)) {
        throw std::runtime_error("OpenQASM 2.0 export only supports control value 1");
    }
    if (op.controls.size() > 1) {
        throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
    }
    std::string name = std::string(op.name);
    std::vector<std::uint64_t> qubits;
    if (op.controls.empty()) {
        qubits = {op.target};
    } else {
        name = "c" + name;
        qubits = {op.controls[0], op.target};
    }
    return {OpenQasm2Op::Kind::Gate, name, {format_angle(op.angle)}, qubits, std::nullopt};
}

// Lower a U-family export record to a concrete OpenQASM 2.0 operation.  QASM2
// only supports controlled forms for u1 and u3 in the qelib1 subset used here.
[[nodiscard]] OpenQasm2Op lower_u_to_qasm2(const UGateOp& op) {
    if (!all_control_values_are_one(op.control_values)) {
        throw std::runtime_error("OpenQASM 2.0 export only supports control value 1");
    }
    if (op.controls.size() > 1) {
        throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
    }
    std::vector<std::string> params;
    params.reserve(op.params.size());
    for (double param : op.params) params.push_back(format_angle(param));
    if (op.controls.empty()) {
        return {OpenQasm2Op::Kind::Gate, std::string(op.name), params, {op.target}, std::nullopt};
    }
    if (op.name == "u1" || op.name == "u3") {
        return {OpenQasm2Op::Kind::Gate,
                "c" + std::string(op.name),
                params,
                {op.controls[0], op.target},
                std::nullopt};
    }
    throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
}

// Convert one dialect-neutral export operation into an OpenQASM 2.0 operation.
// This is where QASM2 expressibility limits are enforced.
[[nodiscard]] std::optional<OpenQasm2Op> lower_to_qasm2(const CircuitOpView& view) {
    return std::visit(
        [](const auto& op) -> std::optional<OpenQasm2Op> {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, IgnoredOp>) {
                return std::nullopt;
            } else if constexpr (std::is_same_v<Op, NamedUnitaryOp>) {
                return lower_named_unitary_to_qasm2(op);
            } else if constexpr (std::is_same_v<Op, RotationOp>) {
                return lower_rotation_to_qasm2(op);
            } else if constexpr (std::is_same_v<Op, UGateOp>) {
                return lower_u_to_qasm2(op);
            } else if constexpr (std::is_same_v<Op, MeasurementOp>) {
                if (op.reset) {
                    throw std::runtime_error("OpenQASM 2.0 export does not support reset-after-measurement gates");
                }
                return OpenQasm2Op{OpenQasm2Op::Kind::Measure, "measure", {}, {op.target}, op.classical_bit};
            } else if constexpr (std::is_same_v<Op, UnsupportedCircuitOp>) {
                throw std::runtime_error(op.reason);
            }
        },
        view);
}

// Render one OpenQasm2Op as a single OpenQASM statement.  The writer only sees
// already-lowered QASM2 operations, not Scaluq gate internals.
[[nodiscard]] std::string write_qasm2_op(const OpenQasm2Op& op) {
    if (op.kind == OpenQasm2Op::Kind::Measure) {
        if (!op.classical_bit) {
            throw std::runtime_error("measure operation is missing a classical bit");
        }
        return "measure " + q(op.qubits[0]) + " -> " + c(*op.classical_bit) + ";";
    }
    std::ostringstream out;
    out << op.name;
    if (!op.params.empty()) {
        out << '(';
        for (std::size_t i = 0; i < op.params.size(); ++i) {
            if (i != 0) out << ", ";
            out << op.params[i];
        }
        out << ')';
    }
    out << ' ';
    for (std::size_t i = 0; i < op.qubits.size(); ++i) {
        if (i != 0) out << ", ";
        out << q(op.qubits[i]);
    }
    out << ';';
    return out.str();
}

}  // namespace

template <Precision Prec>
Qasm2Circuit<Prec> loads(std::string_view source) {
    return Reader<Prec>().read(source);
}

template <Precision Prec>
std::string dumps(const Circuit<Prec>& circuit, std::optional<std::uint64_t> n_qubits) {
    std::uint64_t required_n_qubits = 0;
    std::uint64_t required_n_clbits = 0;
    std::ostringstream body;
    for (const auto& gate_with_key : circuit.gate_list()) {
        CircuitOpView view;
        if (gate_with_key.index() == 0) {
            view = classify_gate_for_export(std::get<0>(gate_with_key));
        } else {
            view = classify_param_gate_for_export(std::get<1>(gate_with_key));
        }
        std::optional<OpenQasm2Op> op = lower_to_qasm2(view);
        if (op) {
            required_n_qubits = std::max(required_n_qubits, required_qubits(*op));
            required_n_clbits = std::max(required_n_clbits, required_clbits(*op));
            body << write_qasm2_op(*op) << '\n';
        }
    }
    if (n_qubits && *n_qubits < required_n_qubits) {
        throw std::runtime_error("specified n_qubits is smaller than circuit operands");
    }
    if (n_qubits.value_or(required_n_qubits) == 0) {
        throw std::runtime_error("OpenQASM 2.0 export requires at least one qubit");
    }
    std::ostringstream out;
    out << "OPENQASM 2.0;\n";
    out << "include \"qelib1.inc\";\n";
    out << "qreg q[" << n_qubits.value_or(required_n_qubits) << "];\n";
    if (required_n_clbits != 0) {
        out << "creg c[" << required_n_clbits << "];\n";
    }
    out << body.str();
    return out.str();
}

template Qasm2Circuit<internal::Prec> loads<internal::Prec>(std::string_view source);
template std::string dumps<internal::Prec>(const Circuit<internal::Prec>& circuit,
                                           std::optional<std::uint64_t> n_qubits);

}  // namespace scaluq::qasm2
