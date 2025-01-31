#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

<<<<<<< HEAD
template <Precision Prec>
=======
template <std::floating_point Fp, ExecutionSpace Sp>
>>>>>>> set-space
class Operator {
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;

public:
    Operator() = default;  // for enable operator= from json
    explicit Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() const { return _n_qubits; }
<<<<<<< HEAD
    [[nodiscard]] inline const std::vector<PauliOperator<Prec>>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator<Prec>& mpt) { add_operator(PauliOperator<Prec>{mpt}); }
    void add_operator(PauliOperator<Prec>&& mpt);
=======
    [[nodiscard]] inline const std::vector<PauliOperator<Fp, Sp>>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator<Fp, Sp>& mpt) {
        add_operator(PauliOperator<Fp, Sp>{mpt});
    }
    void add_operator(PauliOperator<Fp, Sp>&& mpt);
>>>>>>> set-space

    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()());

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

<<<<<<< HEAD
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
=======
    void apply_to_state(StateVector<Fp, Sp>& state_vector) const;

    [[nodiscard]] Complex<Fp> get_expectation_value(const StateVector<Fp, Sp>& state_vector) const;

    [[nodiscard]] Complex<Fp> get_transition_amplitude(
        const StateVector<Fp, Sp>& state_vector_bra,
        const StateVector<Fp, Sp>& state_vector_ket) const;

    // not implemented yet
    [[nodiscard]] Complex<Fp> solve_gound_state_eigenvalue_by_arnoldi_method(
        const StateVector<Fp, Sp>& state, std::uint64_t iter_count, Complex<Fp> mu = 0.) const;
    // not implemented yet
    [[nodiscard]] Complex<Fp> solve_gound_state_eigenvalue_by_power_method(
        const StateVector<Fp, Sp>& state, std::uint64_t iter_count, Complex<Fp> mu = 0.) const;
>>>>>>> set-space

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
<<<<<<< HEAD
    Operator& operator+=(const PauliOperator<Prec>& pauli);
    Operator operator+(const PauliOperator<Prec>& pauli) const { return Operator(*this) += pauli; }
    Operator& operator-=(const PauliOperator<Prec>& pauli) { return *this += pauli * -1.; }
    Operator operator-(const PauliOperator<Prec>& pauli) const { return Operator(*this) -= pauli; }
    Operator& operator*=(const PauliOperator<Prec>& pauli);
    Operator operator*(const PauliOperator<Prec>& pauli) const { return Operator(*this) *= pauli; }
=======
    Operator& operator+=(const PauliOperator<Fp, Sp>& pauli);
    Operator operator+(const PauliOperator<Fp, Sp>& pauli) const {
        return Operator(*this) += pauli;
    }
    Operator& operator-=(const PauliOperator<Fp, Sp>& pauli) { return *this += pauli * -1; }
    Operator operator-(const PauliOperator<Fp, Sp>& pauli) const {
        return Operator(*this) -= pauli;
    }
    Operator& operator*=(const PauliOperator<Fp, Sp>& pauli);
    Operator operator*(const PauliOperator<Fp, Sp>& pauli) const {
        return Operator(*this) *= pauli;
    }
>>>>>>> set-space

    friend void to_json(Json& j, const Operator& op) {
        j = Json{{"n_qubits", op.n_qubits()}, {"terms", Json::array()}};
        for (const auto& pauli : op.terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::uint32_t n = j.at("n_qubits").get<std::uint32_t>();
<<<<<<< HEAD
        Operator<Prec> res(n);
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            StdComplex coef = term.at("coef").get<StdComplex>();
=======
        Operator<Fp, Sp> res(n);
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            Complex<Fp> coef = term.at("coef").get<Complex<Fp>>();
>>>>>>> set-space
            res.add_operator({pauli_string, coef});
        }
        op = res;
    }

private:
<<<<<<< HEAD
    std::vector<PauliOperator<Prec>> _terms;
=======
    std::vector<PauliOperator<Fp, Sp>> _terms;
>>>>>>> set-space
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
<<<<<<< HEAD
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
=======
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator<Fp, Sp>>(m, "Operator", "General quantum operator class.")
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian", &Operator<Fp, Sp>::is_hermitian, "Check if the operator is Hermitian.")
        .def("n_qubits",
             &Operator<Fp, Sp>::n_qubits,
             "Get the number of qubits the operator acts on.")
        .def("terms",
             &Operator<Fp, Sp>::terms,
             "Get the list of Pauli terms that make up the operator.")
        .def(
            "to_string", &Operator<Fp, Sp>::to_string, "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator<Fp, Sp>&>(&Operator<Fp, Sp>::add_operator),
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator<Fp, Sp>& op,
>>>>>>> set-space
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
<<<<<<< HEAD
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
=======
        .def("optimize",
             &Operator<Fp, Sp>::optimize,
             "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator<Fp, Sp>::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state",
             &Operator<Fp, Sp>::apply_to_state,
             "Apply the operator to a state vector.")
        .def("get_expectation_value",
             &Operator<Fp, Sp>::get_expectation_value,
             "Get the expectation value of the operator with respect to a state vector.")
        .def("get_transition_amplitude",
             &Operator<Fp, Sp>::get_transition_amplitude,
>>>>>>> set-space
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
<<<<<<< HEAD
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
=======
        .def(nb::self += PauliOperator<Fp, Sp>())
        .def(nb::self + PauliOperator<Fp, Sp>())
        .def(nb::self -= PauliOperator<Fp, Sp>())
        .def(nb::self - PauliOperator<Fp, Sp>())
        .def(nb::self *= PauliOperator<Fp, Sp>())
        .def(nb::self * PauliOperator<Fp, Sp>())
        .def(
            "to_json",
            [](const Operator<Fp, Sp>& op) { return Json(op).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](Operator<Fp, Sp>& op, const std::string& str) { op = nlohmann::json::parse(str); },
>>>>>>> set-space
            "Read an object from the JSON representation of the operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
