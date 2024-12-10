#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

template <std::floating_point Fp>
class Operator {
public:
    Operator() = default;  // for enable operator= from json
    explicit Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator<Fp>>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator<Fp>& mpt) { add_operator(PauliOperator<Fp>{mpt}); }
    void add_operator(PauliOperator<Fp>&& mpt);

    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()());

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector<Fp>& state_vector) const;

    [[nodiscard]] Complex<Fp> get_expectation_value(const StateVector<Fp>& state_vector) const;

    [[nodiscard]] Complex<Fp> get_transition_amplitude(
        const StateVector<Fp>& state_vector_bra, const StateVector<Fp>& state_vector_ket) const;

    // not implemented yet
    [[nodiscard]] Complex<Fp> solve_gound_state_eigenvalue_by_arnoldi_method(
        const StateVector<Fp>& state, std::uint64_t iter_count, Complex<Fp> mu = 0.) const;
    // not implemented yet
    [[nodiscard]] Complex<Fp> solve_gound_state_eigenvalue_by_power_method(
        const StateVector<Fp>& state, std::uint64_t iter_count, Complex<Fp> mu = 0.) const;

    Operator& operator*=(Complex<Fp> coef);
    Operator operator*(Complex<Fp> coef) const { return Operator(*this) *= coef; }
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1; }
    Operator& operator+=(const Operator& target);
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const;
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator<Fp>& pauli);
    Operator operator+(const PauliOperator<Fp>& pauli) const { return Operator(*this) += pauli; }
    Operator& operator-=(const PauliOperator<Fp>& pauli) { return *this += pauli * -1; }
    Operator operator-(const PauliOperator<Fp>& pauli) const { return Operator(*this) -= pauli; }
    Operator& operator*=(const PauliOperator<Fp>& pauli);
    Operator operator*(const PauliOperator<Fp>& pauli) const { return Operator(*this) *= pauli; }

    friend void to_json(Json& j, const Operator& op) {
        j = Json{{"n_qubits", op.n_qubits()}, {"terms", Json::array()}};
        for (const auto& pauli : op.terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::uint32_t n = j.at("n_qubits").get<std::uint32_t>();
        Operator<Fp> res(n);
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            Kokkos::complex<Fp> coef = term.at("coef").get<Kokkos::complex<Fp>>();
            res.add_operator({pauli_string, coef});
        }
        op = res;
    }

private:
    std::vector<PauliOperator<Fp>> _terms;
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator<double>>(m, "Operator", "General quantum operator class.")
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian", &Operator<double>::is_hermitian, "Check if the operator is Hermitian.")
        .def("n_qubits",
             &Operator<double>::n_qubits,
             "Get the number of qubits the operator acts on.")
        .def("terms",
             &Operator<double>::terms,
             "Get the list of Pauli terms that make up the operator.")
        .def(
            "to_string", &Operator<double>::to_string, "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator<double>&>(&Operator<double>::add_operator),
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator<double>& op,
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
             &Operator<double>::optimize,
             "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator<double>::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state",
             &Operator<double>::apply_to_state,
             "Apply the operator to a state vector.")
        .def("get_expectation_value",
             &Operator<double>::get_expectation_value,
             "Get the expectation value of the operator with respect to a state vector.")
        .def("get_transition_amplitude",
             &Operator<double>::get_transition_amplitude,
             "Get the transition amplitude of the operator between two state vectors.")
        .def(nb::self *= Complex<double>())
        .def(nb::self * Complex<double>())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self += PauliOperator<double>())
        .def(nb::self + PauliOperator<double>())
        .def(nb::self -= PauliOperator<double>())
        .def(nb::self - PauliOperator<double>())
        .def(nb::self *= PauliOperator<double>())
        .def(nb::self * PauliOperator<double>());
}
}  // namespace internal
#endif
}  // namespace scaluq
