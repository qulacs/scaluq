#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class Operator {
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;

public:
    Operator() = default;  // for enable operator= from json
    explicit Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator<Prec, Space>>& terms() const {
        return _terms;
    }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator<Prec, Space>& mpt) {
        add_operator(PauliOperator<Prec, Space>{mpt});
    }
    void add_operator(PauliOperator<Prec, Space>&& mpt);

    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()());

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    [[nodiscard]] ComplexMatrix get_matrix() const;

    void apply_to_state(StateVector<Prec, Space>& state_vector) const;

    [[nodiscard]] StdComplex get_expectation_value(
        const StateVector<Prec, Space>& state_vector) const;
    [[nodiscard]] std::vector<StdComplex> get_expectation_value(
        const StateVectorBatched<Prec, Space>& states) const;

    [[nodiscard]] ComplexMatrix get_matrix_ignoring_coef() const;

    [[nodiscard]] StdComplex get_transition_amplitude(
        const StateVector<Prec, Space>& state_vector_bra,
        const StateVector<Prec, Space>& state_vector_ket) const;
    [[nodiscard]] std::vector<StdComplex> get_transition_amplitude(
        const StateVectorBatched<Prec, Space>& states_bra,
        const StateVectorBatched<Prec, Space>& states_ket) const;

    // not implemented yet
    [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_arnoldi_method(
        const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;
    // not implemented yet
    [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_power_method(
        const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;

    Operator& operator*=(StdComplex coef);
    Operator operator*(StdComplex coef) const { return Operator(*this) *= coef; }
    Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1.; }
    Operator& operator+=(const Operator& target);
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const;
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator<Prec, Space>& pauli);
    Operator operator+(const PauliOperator<Prec, Space>& pauli) const {
        return Operator(*this) += pauli;
    }
    Operator& operator-=(const PauliOperator<Prec, Space>& pauli) { return *this += pauli * -1.; }
    Operator operator-(const PauliOperator<Prec, Space>& pauli) const {
        return Operator(*this) -= pauli;
    }
    Operator& operator*=(const PauliOperator<Prec, Space>& pauli);
    Operator operator*(const PauliOperator<Prec, Space>& pauli) const {
        return Operator(*this) *= pauli;
    }

    friend void to_json(Json& j, const Operator& op) {
        j = Json{{"n_qubits", op.n_qubits()}, {"terms", Json::array()}};
        for (const auto& pauli : op.terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::uint32_t n = j.at("n_qubits").get<std::uint32_t>();
        Operator<Prec, Space> res(n);
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            StdComplex coef = term.at("coef").get<StdComplex>();
            res.add_operator({pauli_string, coef});
        }
        op = res;
    }

private:
    std::vector<PauliOperator<Prec, Space>> _terms;
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator<Prec, Space>>(
        m,
        "Operator",
        DocString()
            .desc("General quantum operator class.")
            .desc("Given `qubit_count: int`, Initialize operator with specified number of qubits.")
            .ex(DocString::Code(
                {">>> pauli = PauliOperator(\"X 3 Y 2\")",
                 ">>> operator = Operator(4)",
                 ">>> operator.add_operator(pauli)",
                 ">>> print(operator.to_json())",
                 "{\"coef\":{\"imag\":0.0,\"real\":1.0},\"pauli_string\":\"X 3 Y 2\"}"}))
            .build_as_google_style()
            .c_str())
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian",
             &Operator<Prec, Space>::is_hermitian,
             "Check if the operator is Hermitian.")
        .def("n_qubits",
             &Operator<Prec, Space>::n_qubits,
             "Get the number of qubits the operator acts on.")
        .def("terms",
             &Operator<Prec, Space>::terms,
             "Get the list of Pauli terms that make up the operator.")
        .def("to_string",
             &Operator<Prec, Space>::to_string,
             "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator<Prec, Space>&>(
                 &Operator<Prec, Space>::add_operator),
             "pauli"_a,
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator<Prec, Space>& op,
               std::uint64_t operator_count,
               std::optional<std::uint64_t> seed) {
                return op.add_random_operator(operator_count,
                                              seed.value_or(std::random_device{}()));
            },
            "operator_count"_a,
            "seed"_a = std::nullopt,
            "Add a specified number of random Pauli operators to this operator. An optional "
            "seed "
            "can be provided for reproducibility.")
        .def("optimize",
             &Operator<Prec, Space>::optimize,
             "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator<Prec, Space>::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state",
             &Operator<Prec, Space>::apply_to_state,
             "state"_a,
             "Apply the operator to a state vector.")
        .def(
            "get_expectation_value",
            [](const Operator<Prec, Space>& op, const StateVector<Prec, Space>& state) {
                return op.get_expectation_value(state);
            },
            "state"_a,
            "Get the expectation value of the operator with respect to a state vector.")
        .def(
            "get_expectation_value",
            [](const Operator<Prec, Space>& op, const StateVectorBatched<Prec, Space>& states) {
                return op.get_expectation_value(states);
            },
            "states"_a,
            "Get the expectation values of the operator for a batch of state vectors.")
        .def(
            "get_transition_amplitude",
            [](const Operator<Prec, Space>& op,
               const StateVector<Prec, Space>& state_source,
               const StateVector<Prec, Space>& state_target) {
                return op.get_transition_amplitude(state_source, state_target);
            },
            "source"_a,
            "target"_a,
            "Get the transition amplitude of the operator between two state vectors.")
        .def(
            "get_transition_amplitude",
            [](const Operator<Prec, Space>& op,
               const StateVectorBatched<Prec, Space>& states_source,
               const StateVectorBatched<Prec, Space>& states_target) {
                return op.get_transition_amplitude(states_source, states_target);
            },
            "states_source"_a,
            "states_target"_a,
            "Get the transition amplitudes of the operator for a batch of state vectors.")
        .def("get_matrix",
             &Operator<Prec, Space>::get_matrix,
             "Get matrix representation of the Operator. Tensor product is applied from "
             "n_qubits-1 to 0.")
        .def(nb::self *= StdComplex())
        .def(nb::self * StdComplex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self += PauliOperator<Prec, Space>())
        .def(nb::self + PauliOperator<Prec, Space>())
        .def(nb::self -= PauliOperator<Prec, Space>())
        .def(nb::self - PauliOperator<Prec, Space>())
        .def(nb::self *= PauliOperator<Prec, Space>())
        .def(nb::self * PauliOperator<Prec, Space>())
        .def(
            "to_json",
            [](const Operator<Prec, Space>& op) { return Json(op).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Operator<Prec, Space>& op, const std::string& str) {
                op = nlohmann::json::parse(str);
            },
            "json_str"_a,
            "Read an object from the JSON representation of the operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
