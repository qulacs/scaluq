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

struct LinearExpr {
    double constant = 0.;
    std::optional<std::string> symbol;
    double coefficient = 0.;
};

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
                if (rhs.symbol || rhs.constant == 0.) {
                    throw std::runtime_error(error("division by non-constant expression"));
                }
                lhs.constant /= rhs.constant;
                lhs.coefficient /= rhs.constant;
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
                return {
                    .constant = std::numbers::pi, .symbol = std::nullopt, .coefficient = 0.};
            }
            return {.constant = 0., .symbol = id, .coefficient = 1.};
        }

        char* end = nullptr;
        std::string rest(_source.substr(_pos));
        double value = std::strtod(rest.c_str(), &end);
        if (end == rest.c_str()) throw std::runtime_error(error("expected number or identifier"));
        _pos += static_cast<std::size_t>(end - rest.c_str());
        return {.constant = value, .symbol = std::nullopt, .coefficient = 0.};
    }

    [[nodiscard]] LinearExpr add(LinearExpr lhs, const LinearExpr& rhs) const {
        lhs.constant += rhs.constant;
        if (lhs.symbol && rhs.symbol && *lhs.symbol != *rhs.symbol) {
            throw std::runtime_error(error("multiple symbolic parameters in one angle are unsupported"));
        }
        if (!lhs.symbol && rhs.symbol) lhs.symbol = rhs.symbol;
        lhs.coefficient += rhs.coefficient;
        return lhs;
    }

    [[nodiscard]] LinearExpr negate(LinearExpr expr) const {
        expr.constant = -expr.constant;
        expr.coefficient = -expr.coefficient;
        return expr;
    }

    [[nodiscard]] LinearExpr multiply(const LinearExpr& lhs, const LinearExpr& rhs) const {
        if (lhs.symbol && rhs.symbol) {
            throw std::runtime_error(error("non-linear symbolic angle is unsupported"));
        }
        if (lhs.symbol) {
            return {.constant = lhs.constant * rhs.constant,
                    .symbol = lhs.symbol,
                    .coefficient = lhs.coefficient * rhs.constant};
        }
        if (rhs.symbol) {
            return {.constant = lhs.constant * rhs.constant,
                    .symbol = rhs.symbol,
                    .coefficient = lhs.constant * rhs.coefficient};
        }
        return {.constant = lhs.constant * rhs.constant,
                .symbol = std::nullopt,
                .coefficient = 0.};
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

[[nodiscard]] bool all_control_values_are_one(const Json& j) {
    if (!j.contains("control_value")) return true;
    for (std::uint64_t value : j.at("control_value").get<std::vector<std::uint64_t>>()) {
        if (value != 1) return false;
    }
    return true;
}

[[nodiscard]] std::string format_angle(double value) {
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

[[nodiscard]] std::string format_param_angle(std::string_view key, double coef) {
    if (coef == 1.) return std::string(key);
    if (coef == -1.) return "-" + std::string(key);
    return format_angle(coef) + "*" + std::string(key);
}

template <Precision Prec>
void add_rotation(Circuit<Prec>& circuit,
                  std::string_view name,
                  std::uint64_t target,
                  const LinearExpr& angle,
                  const std::vector<std::uint64_t>& controls,
                  Location loc) {
    if (angle.symbol) {
        if (angle.constant != 0.) {
            throw std::runtime_error(
                make_error(loc, "symbolic angles with a constant offset are unsupported"));
        }
        if (name == "rx") {
            circuit.add_param_gate(
                gate::ParamRX<Prec>(target, angle.coefficient, controls), *angle.symbol);
        } else if (name == "ry") {
            circuit.add_param_gate(
                gate::ParamRY<Prec>(target, angle.coefficient, controls), *angle.symbol);
        } else if (name == "rz") {
            circuit.add_param_gate(
                gate::ParamRZ<Prec>(target, angle.coefficient, controls), *angle.symbol);
        } else {
            throw std::runtime_error(
                make_error(loc, "symbolic angles are only supported for rx, ry, and rz"));
        }
        return;
    }
    if (name == "rx")
        circuit.add_gate(gate::RX<Prec>(target, angle.constant, controls));
    else if (name == "ry")
        circuit.add_gate(gate::RY<Prec>(target, angle.constant, controls));
    else if (name == "rz")
        circuit.add_gate(gate::RZ<Prec>(target, angle.constant, controls));
}

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

    void handle_statement(const Statement& statement) {
        std::string stmt = trim(statement.text);
        std::string low = lower(stmt);
        if (low == "openqasm 2.0") {
            _saw_version = true;
            return;
        }
        if (starts_with_word(low, "include")) {
            if (low != "include \"qelib1.inc\"") {
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
        for (std::string_view unsupported : {"measure", "reset", "if", "gate", "opaque"}) {
            if (starts_with_word(low, unsupported)) {
                throw std::runtime_error(
                    make_error(statement.loc, std::string(unsupported) + " is not supported yet"));
            }
        }
        parse_gate(stmt, statement.loc);
    }

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
        if (name == "cx") name = "cx";

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
            if (params[0].symbol) throw std::runtime_error(make_error(loc, "symbolic u1 is unsupported"));
            _result.circuit.add_gate(gate::U1<Prec>(qubits[0], params[0].constant));
            return;
        }
        if (name == "u2" || name == "u3") {
            expect(name, params, qubits, name == "u2" ? 2 : 3, 1, loc);
            for (const auto& param : params) {
                if (param.symbol) throw std::runtime_error(make_error(loc, "symbolic u2/u3 is unsupported"));
            }
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
            if (params[0].symbol) throw std::runtime_error(make_error(loc, "symbolic cu1 is unsupported"));
            _result.circuit.add_gate(gate::U1<Prec>(qubits[1], params[0].constant, {qubits[0]}));
            return;
        }
        if (name == "cu3") {
            expect(name, params, qubits, 3, 2, loc);
            for (const auto& param : params) {
                if (param.symbol) throw std::runtime_error(make_error(loc, "symbolic cu3 is unsupported"));
            }
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
};

[[nodiscard]] std::uint64_t update_required_qubits(std::uint64_t current, const Json& gate_json) {
    auto update = [&current](const Json& values) {
        for (std::uint64_t q : values.get<std::vector<std::uint64_t>>()) {
            current = std::max(current, q + 1);
        }
    };
    if (gate_json.contains("target")) update(gate_json.at("target"));
    if (gate_json.contains("control")) update(gate_json.at("control"));
    return current;
}

template <Precision Prec>
std::string dump_gate(const Gate<Prec>& gate) {
    Json j = gate;
    if (!all_control_values_are_one(j)) {
        throw std::runtime_error("OpenQASM 2.0 export only supports control value 1");
    }
    std::string type = j.at("type").get<std::string>();
    std::vector<std::uint64_t> targets =
        j.contains("target") ? j.at("target").get<std::vector<std::uint64_t>>() : std::vector<std::uint64_t>{};
    std::vector<std::uint64_t> controls =
        j.contains("control") ? j.at("control").get<std::vector<std::uint64_t>>() : std::vector<std::uint64_t>{};
    auto q = [](std::uint64_t index) { return "q[" + std::to_string(index) + "]"; };

    if (type == "I") return "";
    if (type == "X" && controls.size() == 1) return "cx " + q(controls[0]) + ", " + q(targets[0]) + ";";
    if (type == "Z" && controls.size() == 1) return "cz " + q(controls[0]) + ", " + q(targets[0]) + ";";
    if (type == "X" && controls.size() == 2) {
        return "ccx " + q(controls[0]) + ", " + q(controls[1]) + ", " + q(targets[0]) + ";";
    }
    if (type == "Swap" && controls.size() == 1) {
        return "cswap " + q(controls[0]) + ", " + q(targets[0]) + ", " + q(targets[1]) + ";";
    }
    if (controls.size() > 1) throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");

    std::string prefix;
    if (controls.size() == 1) {
        if (type == "Y") prefix = "cy " + q(controls[0]) + ", ";
        else if (type == "H") prefix = "ch " + q(controls[0]) + ", ";
        else if (type == "RX") prefix = "crx(" + format_angle(j.at("angle").get<double>()) + ") " + q(controls[0]) + ", ";
        else if (type == "RY") prefix = "cry(" + format_angle(j.at("angle").get<double>()) + ") " + q(controls[0]) + ", ";
        else if (type == "RZ") prefix = "crz(" + format_angle(j.at("angle").get<double>()) + ") " + q(controls[0]) + ", ";
        else if (type == "U1") prefix = "cu1(" + format_angle(j.at("lambda").get<double>()) + ") " + q(controls[0]) + ", ";
        else if (type == "U3")
            prefix = "cu3(" + format_angle(j.at("theta").get<double>()) + ", " +
                     format_angle(j.at("phi").get<double>()) + ", " +
                     format_angle(j.at("lambda").get<double>()) + ") " + q(controls[0]) + ", ";
        else
            throw std::runtime_error("unsupported controlled gate for OpenQASM 2.0 export");
        return prefix + q(targets[0]) + ";";
    }

    if (type == "X" || type == "Y" || type == "Z" || type == "H" || type == "S" || type == "T") {
        return lower(type) + " " + q(targets[0]) + ";";
    }
    if (type == "Sdag") return "sdg " + q(targets[0]) + ";";
    if (type == "Tdag") return "tdg " + q(targets[0]) + ";";
    if (type == "RX") return "rx(" + format_angle(j.at("angle").get<double>()) + ") " + q(targets[0]) + ";";
    if (type == "RY") return "ry(" + format_angle(j.at("angle").get<double>()) + ") " + q(targets[0]) + ";";
    if (type == "RZ") return "rz(" + format_angle(j.at("angle").get<double>()) + ") " + q(targets[0]) + ";";
    if (type == "U1") return "u1(" + format_angle(j.at("lambda").get<double>()) + ") " + q(targets[0]) + ";";
    if (type == "U2") {
        return "u2(" + format_angle(j.at("phi").get<double>()) + ", " +
               format_angle(j.at("lambda").get<double>()) + ") " + q(targets[0]) + ";";
    }
    if (type == "U3") {
        return "u3(" + format_angle(j.at("theta").get<double>()) + ", " +
               format_angle(j.at("phi").get<double>()) + ", " +
               format_angle(j.at("lambda").get<double>()) + ") " + q(targets[0]) + ";";
    }
    if (type == "Swap") return "swap " + q(targets[0]) + ", " + q(targets[1]) + ";";
    throw std::runtime_error("unsupported gate for OpenQASM 2.0 export: " + type);
}

template <Precision Prec>
std::string dump_param_gate(const std::pair<ParamGate<Prec>, std::string>& gate_with_key) {
    Json j = gate_with_key.first;
    if (!all_control_values_are_one(j)) {
        throw std::runtime_error("OpenQASM 2.0 export only supports control value 1");
    }
    std::string type = j.at("type").get<std::string>();
    std::vector<std::uint64_t> targets = j.at("target").get<std::vector<std::uint64_t>>();
    std::vector<std::uint64_t> controls =
        j.contains("control") ? j.at("control").get<std::vector<std::uint64_t>>() : std::vector<std::uint64_t>{};
    if (controls.size() > 1) throw std::runtime_error("unsupported controlled param gate for OpenQASM 2.0 export");
    auto q = [](std::uint64_t index) { return "q[" + std::to_string(index) + "]"; };
    std::string name;
    if (type == "ParamRX") name = controls.empty() ? "rx" : "crx";
    if (type == "ParamRY") name = controls.empty() ? "ry" : "cry";
    if (type == "ParamRZ") name = controls.empty() ? "rz" : "crz";
    if (name.empty()) throw std::runtime_error("unsupported param gate for OpenQASM 2.0 export: " + type);
    std::string angle = format_param_angle(gate_with_key.second, j.at("param_coef").get<double>());
    if (controls.empty()) return name + "(" + angle + ") " + q(targets[0]) + ";";
    return name + "(" + angle + ") " + q(controls[0]) + ", " + q(targets[0]) + ";";
}

}  // namespace

template <Precision Prec>
Qasm2Circuit<Prec> loads(std::string_view source) {
    return Reader<Prec>().read(source);
}

template <Precision Prec>
std::string dumps(const Circuit<Prec>& circuit, std::optional<std::uint64_t> n_qubits) {
    std::uint64_t required_n_qubits = 0;
    std::ostringstream body;
    for (const auto& gate_with_key : circuit.gate_list()) {
        std::string line;
        if (gate_with_key.index() == 0) {
            Json j = std::get<0>(gate_with_key);
            required_n_qubits = update_required_qubits(required_n_qubits, j);
            line = dump_gate(std::get<0>(gate_with_key));
        } else {
            Json j = std::get<1>(gate_with_key).first;
            required_n_qubits = update_required_qubits(required_n_qubits, j);
            line = dump_param_gate(std::get<1>(gate_with_key));
        }
        if (!line.empty()) body << line << '\n';
    }
    if (n_qubits && *n_qubits < required_n_qubits) {
        throw std::runtime_error("specified n_qubits is smaller than circuit operands");
    }
    std::ostringstream out;
    out << "OPENQASM 2.0;\n";
    out << "include \"qelib1.inc\";\n";
    out << "qreg q[" << n_qubits.value_or(required_n_qubits) << "];\n";
    out << body.str();
    return out.str();
}

template Qasm2Circuit<internal::Prec> loads<internal::Prec>(std::string_view source);
template std::string dumps<internal::Prec>(const Circuit<internal::Prec>& circuit,
                                           std::optional<std::uint64_t> n_qubits);

}  // namespace scaluq::qasm2
