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
    using ExecutionSpaceType = internal::SpaceType<Space>;

public:
    Operator() = default;
    explicit Operator(std::uint64_t n_terms) : _terms("terms", n_terms) {}
    explicit Operator(std::vector<PauliOperator<Prec, Space>> terms);

    [[nodiscard]] Operator copy() const;
    void load(const std::vector<PauliOperator<Prec, Space>>& terms);
    static Operator uninitialized_operator(std::uint64_t n_terms);

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::vector<PauliOperator<Prec, Space>> get_terms() const {
        return internal::convert_view_to_vector<PauliOperator<Prec, Space>, Space>(_terms);
    }
    [[nodiscard]] std::string to_string() const;

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    [[nodiscard]] ComplexMatrix get_full_matrix(std::uint64_t n_qubits) const;

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

    Operator operator*(StdComplex coef) const;
    Operator& operator*=(StdComplex coef);
    Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1.; }
    Operator operator+(const Operator& target) const;
    Operator operator-(const Operator& target) const { return *this + target * -1.; }
    Operator operator*(const Operator& target) const;
    Operator operator+(const PauliOperator<Prec, Space>& pauli) const;
    Operator operator-(const PauliOperator<Prec, Space>& pauli) const {
        return *this + pauli * -1.;
    }
    Operator operator*(const PauliOperator<Prec, Space>& pauli) const;
    Operator& operator*=(const PauliOperator<Prec, Space>& pauli);

    friend void to_json(Json& j, const Operator& op) {
        j.clear();
        j["terms"] = Json::array();
        for (const auto& pauli : op.get_terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::vector<PauliOperator<Prec, Space>> res;
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            StdComplex coef = term.at("coef").get<StdComplex>();
            res.emplace_back(pauli_string, coef);
        }
        op = Operator<Prec, Space>(res);
    }

    friend std::ostream& operator<<(std::ostream& os, const Operator& op) {
        return os << op.to_string();
    }

private:
    Kokkos::View<PauliOperator<Prec, Space>*, ExecutionSpaceType> _terms;
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
                {">>> terms = [PauliOperator(\"X 0 Y 2\"), PauliOperator(\"Z 1 X 3\", 2j)]",
                 ">>> op = Operator(terms)",
                 ">>> pauli = PauliOperator(\"X 1\", -1j)",
                 ">>> op *= pauli",
                 ">>> print(op.to_json())",
                 R"({"terms":[{"coef":{"imag":-1.0,"real":0.0},"pauli_string":"X 0 X 1 Y 2"},{"coef":{"imag":2.0,"real":0.0},"pauli_string":"Y 1 X 3"}]})"}))
            .build_as_google_style()
            .c_str())
        .def(nb::init<std::uint64_t>(),
             "n_terms"_a,
             "Initialize operator with specified number of terms.")
        .def(nb::init<std::vector<PauliOperator<Prec, Space>>>(),
             "terms"_a,
             "Initialize operator with given list of terms.")
        .def("is_hermitian",
             &Operator<Prec, Space>::is_hermitian,
             "Check if the operator is Hermitian.")
        .def("load",
             &Operator<Prec, Space>::load,
             "terms"_a,
             "Load the operator with a list of Pauli operators.")
        .def_static("uninitialized_operator",
                    &Operator<Prec, Space>::uninitialized_operator,
                    "n_terms"_a,
                    "Create an uninitialized operator with a specified number of terms.")
        .def("get_terms",
             &Operator<Prec, Space>::get_terms,
             "Get the list of Pauli terms that make up the operator.")
        .def("to_string",
             &Operator<Prec, Space>::to_string,
             "Get string representation of the operator.")
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
             &Operator<Prec, Space>::get_full_matrix,
             "Get matrix representation of the Operator. Tensor product is applied from "
             "n_qubits-1 to 0.")
        .def(nb::self *= StdComplex())
        .def(nb::self * StdComplex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self + PauliOperator<Prec, Space>())
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
            "Read an object from the JSON representation of the operator.")
        .def("to_string",
             &Operator<Prec, Space>::to_string,
             "Get string representation of the operator.")
        .def("__str__",
             &Operator<Prec, Space>::to_string,
             "Get string representation of the operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
