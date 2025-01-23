#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

template <Precision Prec>
class Operator {
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;

public:
    Operator() = default;  // for enable operator= from json
    explicit Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator<Prec>>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator<Prec>& mpt) { add_operator(PauliOperator<Prec>{mpt}); }
    void add_operator(PauliOperator<Prec>&& mpt);

    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()());

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector<Prec>& state_vector) const;

    [[nodiscard]] StdComplex get_expectation_value(const StateVector<Prec>& state_vector) const;

    [[nodiscard]] StdComplex get_transition_amplitude(
        const StateVector<Prec>& state_vector_bra, const StateVector<Prec>& state_vector_ket) const;

    // not implemented yet
    [[nodiscard]] StdComplex solve_gound_state_eigenvalue_by_arnoldi_method(
        const StateVector<Prec>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;
    // not implemented yet
    [[nodiscard]] StdComplex solve_gound_state_eigenvalue_by_power_method(
        const StateVector<Prec>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;

    Operator& operator*=(StdComplex coef);
    Operator operator*(StdComplex coef) const { return Operator(*this) *= coef; }
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1.; }
    Operator& operator+=(const Operator& target);
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const;
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator<Prec>& pauli);
    Operator operator+(const PauliOperator<Prec>& pauli) const { return Operator(*this) += pauli; }
    Operator& operator-=(const PauliOperator<Prec>& pauli) { return *this += pauli * -1.; }
    Operator operator-(const PauliOperator<Prec>& pauli) const { return Operator(*this) -= pauli; }
    Operator& operator*=(const PauliOperator<Prec>& pauli);
    Operator operator*(const PauliOperator<Prec>& pauli) const { return Operator(*this) *= pauli; }

    friend void to_json(Json& j, const Operator& op) {
        j = Json{{"n_qubits", op.n_qubits()}, {"terms", Json::array()}};
        for (const auto& pauli : op.terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::uint32_t n = j.at("n_qubits").get<std::uint32_t>();
        Operator<Prec> res(n);
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            StdComplex coef = term.at("coef").get<StdComplex>();
            res.add_operator({pauli_string, coef});
        }
        op = res;
    }

private:
    std::vector<PauliOperator<Prec>> _terms;
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator<Prec>>(m, "Operator", "General quantum operator class.")
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian", &Operator<Prec>::is_hermitian, "Check if the operator is Hermitian.")
        .def(
            "n_qubits", &Operator<Prec>::n_qubits, "Get the number of qubits the operator acts on.")
        .def("terms",
             &Operator<Prec>::terms,
             "Get the list of Pauli terms that make up the operator.")
        .def("to_string", &Operator<Prec>::to_string, "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator<Prec>&>(&Operator<Prec>::add_operator),
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator<Prec>& op,
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
        .def(
            "optimize", &Operator<Prec>::optimize, "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator<Prec>::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state",
             &Operator<Prec>::apply_to_state,
             "Apply the operator to a state vector.")
        .def("get_expectation_value",
             &Operator<Prec>::get_expectation_value,
             "Get the expectation value of the operator with respect to a state vector.")
        .def("get_transition_amplitude",
             &Operator<Prec>::get_transition_amplitude,
             "Get the transition amplitude of the operator between two state vectors.")
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
        .def(nb::self += PauliOperator<Prec>())
        .def(nb::self + PauliOperator<Prec>())
        .def(nb::self -= PauliOperator<Prec>())
        .def(nb::self - PauliOperator<Prec>())
        .def(nb::self *= PauliOperator<Prec>())
        .def(nb::self * PauliOperator<Prec>())
        .def(
            "to_json",
            [](const Operator<Prec>& op) { return Json(op).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Operator<Prec>& op, const std::string& str) { op = nlohmann::json::parse(str); },
            "Read an object from the JSON representation of the operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
