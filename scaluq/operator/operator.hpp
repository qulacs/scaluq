#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {
class Operator {
public:
    explicit Operator(std::uint64_t n_qubits);

    [[nodiscard]] inline bool is_hermitian() { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator& mpt);
    void add_operator(PauliOperator&& mpt);
    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()());

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_expectation_value_loop(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_arnoldi_method(const StateVector& state,
                                                                         std::uint64_t iter_count,
                                                                         Complex mu = 0.) const;
    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_power_method(const StateVector& state,
                                                                       std::uint64_t iter_count,
                                                                       Complex mu = 0.) const;

    Operator& operator*=(Complex coef);
    Operator operator*(Complex coef) const { return Operator(*this) *= coef; }
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1; }
    Operator& operator+=(const Operator& target);
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const;
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator& pauli);
    Operator operator+(const PauliOperator& pauli) const { return Operator(*this) += pauli; }
    Operator& operator-=(const PauliOperator& pauli) { return *this += pauli * -1; }
    Operator operator-(const PauliOperator& pauli) const { return Operator(*this) -= pauli; }
    Operator& operator*=(const PauliOperator& pauli);
    Operator operator*(const PauliOperator& pauli) const { return Operator(*this) *= pauli; }

private:
    std::vector<PauliOperator> _terms;
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator>(m, "Operator", "General quantum operator class.")
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian", &Operator::is_hermitian, "Check if the operator is Hermitian.")
        .def("n_qubits", &Operator::n_qubits, "Get the number of qubits the operator acts on.")
        .def("terms", &Operator::terms, "Get the list of Pauli terms that make up the operator.")
        .def("to_string", &Operator::to_string, "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator&>(&Operator::add_operator),
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator& op, std::uint64_t operator_count, std::optional<std::uint64_t> seed) {
                return op.add_random_operator(operator_count,
                                              seed.value_or(std::random_device{}()));
            },
            "operator_count"_a,
            "seed"_a = std::nullopt,
            "Add a specified number of random Pauli operators to this operator. An optional seed "
            "can be provided for reproducibility.")
        .def("optimize", &Operator::optimize, "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state", &Operator::apply_to_state, "Apply the operator to a state vector.")
        .def("get_expectation_value",
             &Operator::get_expectation_value,
             "Get the expectation value of the operator with respect to a state vector.")
        .def("get_transition_amplitude",
             &Operator::get_transition_amplitude,
             "Get the transition amplitude of the operator between two state vectors.")
        .def(nb::self *= Complex())
        .def(nb::self * Complex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self += PauliOperator())
        .def(nb::self + PauliOperator())
        .def(nb::self -= PauliOperator())
        .def(nb::self - PauliOperator())
        .def(nb::self *= PauliOperator())
        .def(nb::self * PauliOperator());
}
}  // namespace internal
#endif
}  // namespace scaluq
