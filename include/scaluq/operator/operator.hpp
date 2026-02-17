#pragma once

#include <random>
#include <vector>

#include "../kokkos.hpp"
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
    struct GroundState {
        StdComplex eigenvalue;
        StateVector<Prec, Space> state;
    };

    Operator() = default;
    explicit Operator(std::uint64_t n_terms) : _terms("terms", n_terms) {}
    explicit Operator(const ConcurrentStream& stream)
        : _space(stream.get<ExecutionSpaceType>()) {}
    Operator(const ConcurrentStream& stream, std::uint64_t n_terms)
        : _space(stream.get<ExecutionSpaceType>()), _terms("terms", n_terms) {}
    explicit Operator(std::vector<PauliOperator<Prec>> terms);
    Operator(const ConcurrentStream& stream, std::vector<PauliOperator<Prec>> terms);

    [[nodiscard]] Operator copy() const;
    [[nodiscard]] Operator copy(const ConcurrentStream& stream) const;
    void load(const std::vector<PauliOperator<Prec>>& terms);
    void load(const std::vector<PauliOperator<Prec>>& terms, const ConcurrentStream& stream);
    static Operator uninitialized_operator(std::uint64_t n_terms);
    static Operator uninitialized_operator(const ConcurrentStream& stream, std::uint64_t n_terms);

    [[nodiscard]] inline bool is_hermitian() const { return _is_hermitian; }
    [[nodiscard]] inline std::vector<PauliOperator<Prec>> get_terms() const {
        return internal::convert_view_to_vector<PauliOperator<Prec>, Space>(_terms);
    }
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::uint64_t n_terms() const { return _terms.extent(0); }

    [[nodiscard]] const ExecutionSpaceType& execution_space() const { return _space; }
    [[nodiscard]] ConcurrentStream concurrent_stream() const { return ConcurrentStream(_space); }
    void set_concurrent_stream(const ConcurrentStream& stream) {
        _space = stream.get<ExecutionSpaceType>();
    }

    void optimize();

    [[nodiscard]] Operator get_dagger() const;
    [[nodiscard]] Operator get_dagger(const ConcurrentStream& stream) const;

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
    [[nodiscard]] GroundState solve_ground_state_by_power_method(
        const StateVector<Prec, Space>& initial_state,
        std::uint64_t iter_count,
        std::optional<StdComplex> mu = std::nullopt) const;
    // not implemented yet
    [[nodiscard]] GroundState solve_ground_state_by_arnoldi_method(
        const StateVector<Prec, Space>& initial_state,
        std::uint64_t iter_count,
        std::optional<StdComplex> mu = std::nullopt) const;

    Operator operator*(StdComplex coef) const;
    Operator& operator*=(StdComplex coef);
    Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1.; }
    Operator operator+(const Operator& target) const;
    Operator operator-(const Operator& target) const { return *this + target * -1.; }
    Operator operator*(const Operator& target) const;
    Operator operator+(const PauliOperator<Prec>& pauli) const;
    Operator operator-(const PauliOperator<Prec>& pauli) const { return *this + pauli * -1.; }
    Operator operator*(const PauliOperator<Prec>& pauli) const;
    Operator& operator*=(const PauliOperator<Prec>& pauli);

    friend void to_json(Json& j, const Operator& op) {
        j.clear();
        j["terms"] = Json::array();
        for (const auto& pauli : op.get_terms()) {
            Json tmp = pauli;
            j["terms"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, Operator& op) {
        std::vector<PauliOperator<Prec>> res;
        for (const auto& term : j.at("terms")) {
            std::string pauli_string = term.at("pauli_string").get<std::string>();
            StdComplex coef = term.at("coef").get<StdComplex>();
            res.emplace_back(pauli_string, coef);
        }
        op = Operator<Prec, Space>(res);
    }

    Operator<Prec, ExecutionSpace::Default> to_default_space() const;
    Operator<Prec, ExecutionSpace::Host> to_host_space() const;

    friend std::ostream& operator<<(std::ostream& os, const Operator& op) {
        return os << op.to_string();
    }

    StdComplex calculate_default_mu() const;

    ExecutionSpaceType _space{};
    Kokkos::View<PauliOperator<Prec>*, ExecutionSpaceType> _terms;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator<Prec, Space>> op(
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
            .c_str());
    nb::class_<typename Operator<Prec, Space>::GroundState> ground_state(
        op,
        "GroundState",
        DocString()
            .desc("Structure to hold the ground state information.")
            .build_as_google_style()
            .c_str());
    op.def(nb::init<>(), "Initialize operator with default execution space.")
        .def(nb::init<const ConcurrentStream&>(),
             "stream"_a,
             "Initialize operator with specified execution space.")
        .def(nb::init<std::uint64_t>(),
             "n_terms"_a,
             "Initialize operator with specified number of terms.")
        .def(nb::init<const ConcurrentStream&, std::uint64_t>(),
             "stream"_a,
             "n_terms"_a,
             "Initialize operator with specified execution space and number of terms.")
        .def(nb::init<std::vector<PauliOperator<Prec>>>(),
             "terms"_a,
             "Initialize operator with given list of terms.")
        .def(nb::init<const ConcurrentStream&, std::vector<PauliOperator<Prec>>>(),
             "stream"_a,
             "terms"_a,
             "Initialize operator with specified execution space and given list of terms.")
        .def("copy",
             nb::overload_cast<>(&Operator<Prec, Space>::copy, nb::const_),
             DocString().desc("Return a copy.").build_as_google_style().c_str())
        .def("copy",
             nb::overload_cast<const ConcurrentStream&>(&Operator<Prec, Space>::copy, nb::const_),
             "stream"_a,
             DocString()
                 .desc("Return a copy using the specified execution space.")
                 .arg("stream", "ConcurrentStream", "Execution space instance")
                 .build_as_google_style()
                 .c_str())
        .def("is_hermitian",
             &Operator<Prec, Space>::is_hermitian,
             "Check if the operator is Hermitian.")
        .def("concurrent_stream",
             &Operator<Prec, Space>::concurrent_stream,
             DocString()
                 .desc("Return execution space instance for subsequent operations.")
                 .ret("ConcurrentStream", "execution space instance")
                 .build_as_google_style()
                 .c_str())
        .def("set_concurrent_stream",
             &Operator<Prec, Space>::set_concurrent_stream,
             "stream"_a,
             DocString()
                 .desc("Set execution space instance for subsequent operations.")
                 .arg("stream", "ConcurrentStream", "execution space instance")
                 .build_as_google_style()
                 .c_str())
        .def("load",
             &Operator<Prec, Space>::load,
             "terms"_a,
             "Load the operator with a list of Pauli operators.")
        .def("load",
             nb::overload_cast<const std::vector<PauliOperator<Prec>>&, const ConcurrentStream&>(
                 &Operator<Prec, Space>::load),
             "terms"_a,
             "stream"_a,
             DocString()
                 .desc("Load the operator with a list of Pauli operators.")
                 .arg("terms", "A vector of Pauli operators.")
                 .arg("stream", "ConcurrentStream", "Execution space instance")
                 .build_as_google_style()
                 .c_str())
        .def_static("uninitialized_operator",
                    &Operator<Prec, Space>::uninitialized_operator,
                    "n_terms"_a,
                    "Create an uninitialized operator with a specified number of terms.")
        .def_static("uninitialized_operator",
                    nb::overload_cast<const ConcurrentStream&, std::uint64_t>(
                        &Operator<Prec, Space>::uninitialized_operator),
                    "stream"_a,
                    "n_terms"_a,
                    "Create an uninitialized operator with a specified execution space and number "
                    "of terms.")
        .def("n_terms", &Operator<Prec, Space>::n_terms, "Get the number of terms in the operator.")
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
        .def("get_dagger",
             nb::overload_cast<const ConcurrentStream&>(&Operator<Prec, Space>::get_dagger,
                                                        nb::const_),
             "stream"_a,
             DocString()
                 .desc("Get the adjoint (Hermitian conjugate) using the specified execution space.")
                 .arg("stream", "ConcurrentStream", "Execution space instance")
                 .build_as_google_style()
                 .c_str())
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
        .def(
            "solve_ground_state_by_power_method",
            &Operator<Prec, Space>::solve_ground_state_by_power_method,
            "initial_state"_a,
            "iter_count"_a,
            "mu"_a = std::nullopt,
            DocString()
                .desc("Solve for the ground state using the power method.")
                .arg("initial_state",
                     ":class:`StateVector`",
                     "Initial state vector for the iteration. Passing Haar_random_state is often "
                     "nice.")
                .arg("iter_count", "int", "Number of iterations to perform.")
                .arg("mu",
                     "complex",
                     true,
                     "Shift value to accelerate convergence. The value must be larger than largest "
                     "eigenvalue. If not provided, a default value is calculated internally.")
                .ret(":class:`Operator.GroundState`",
                     "The ground state information including eigenvalue and ground state vector.")
                .ex(DocString::Code(
                    {">>> terms = [PauliOperator(\"\", -3.8505), PauliOperator(\"X 1\", -0.2288), "
                     "PauliOperator(\"Z 1\", -1.0466), PauliOperator(\"X 0\", -0.2288), "
                     "PauliOperator(\"X 0 X 1\", 0.2613), PauliOperator(\"X 0 Z 1\", 0.2288), "
                     "PauliOperator(\"Z 0\", -1.0466), PauliOperator(\"Z 0 X 1\", 0.2288), "
                     "PauliOperator(\"Z 0 Z 1\", 0.2356)]",
                     ">>> op = Operator(terms)",
                     ">>> op *= .5",
                     ">>> ground_state = "
                     "op.solve_ground_state_by_power_method(StateVector.Haar_random_state(2, "
                     "seed=0), 200)",
                     ">>> ground_state.eigenvalue.real # doctest: +ELLIPSIS",
                     "-2.862620764...",
                     ">>> print(ground_state.state.get_amplitudes()) # doctest: +SKIP",
                     "[(0.593378420...-0.801949191...j), "
                     "(-0.00937288681...+0.0126674290...j), "
                     "(-0.00937288706...+0.0126674292...j), "
                     "(-0.0389261878...+0.0526086285...j)]"}))
                .build_as_google_style()
                .c_str())
        .def(
            "solve_ground_state_by_arnoldi_method",
            &Operator<Prec, Space>::solve_ground_state_by_arnoldi_method,
            "initial_state"_a,
            "iter_count"_a,
            "mu"_a = std::nullopt,
            DocString()
                .desc("Solve for the ground state using the Arnoldi method.")
                .arg("initial_state",
                     ":class:`StateVector`",
                     "Initial state vector for the iteration. Passing Haar_random_state is often "
                     "nice.")
                .arg("iter_count",
                     "int",
                     "Number of iterations to perform. Too large value causes error on calculating "
                     "eigenvalue of Hessenberg matrix.")
                .arg("mu",
                     "complex",
                     true,
                     "Shift value to accelerate convergence. The value must be larger than largest "
                     "eigenvalue. If not provided, a default value is calculated internally.")
                .ret(":class:`Operator.GroundState`",
                     "The ground state information including eigenvalue and ground state vector.")
                .ex(DocString::Code(
                    {">>> terms = [PauliOperator(\"\", -3.8505), PauliOperator(\"X 1\", -0.2288), "
                     "PauliOperator(\"Z 1\", -1.0466), PauliOperator(\"X 0\", -0.2288), "
                     "PauliOperator(\"X 0 X 1\", 0.2613), PauliOperator(\"X 0 Z 1\", 0.2288), "
                     "PauliOperator(\"Z 0\", -1.0466), PauliOperator(\"Z 0 X 1\", 0.2288), "
                     "PauliOperator(\"Z 0 Z 1\", 0.2356)]",
                     ">>> op = Operator(terms)",
                     ">>> op *= .5",
                     ">>> ground_state = "
                     "op.solve_ground_state_by_arnoldi_method(StateVector.Haar_random_state(2, "
                     "seed=0), "
                     "40)",
                     ">>> ground_state.eigenvalue.real # doctest: +ELLIPSIS",
                     "-2.862620764...",
                     ">>> ground_state.state.get_amplitudes() # doctest: +SKIP",
                     "[(0.593378420...-0.801949191...j), "
                     "(-0.00937288693...+0.0126674291...j), "
                     "(-0.00937288693...+0.0126674291...j), "
                     "(-0.0389261878...+0.0526086285...j)]"}))
                .build_as_google_style()
                .c_str())
        .def("calculate_default_mu",
             &Operator<Prec, Space>::calculate_default_mu,
             DocString()
                 .desc("Calculate a default shift value `mu` for ground state solvers.")
                 .ret("complex", "Calculated shift value `mu`.")
                 .build_as_google_style()
                 .c_str())
        .def(nb::self *= StdComplex())
        .def(nb::self * StdComplex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self + PauliOperator<Prec>())
        .def(nb::self - PauliOperator<Prec>())
        .def(nb::self *= PauliOperator<Prec>())
        .def(nb::self * PauliOperator<Prec>())
        .def(
            "__iadd__",
            [](Operator<Prec, Space>&, const Operator<Prec, Space>&) -> Operator<Prec, Space>& {
                throw nb::type_error("In-place addition (+=) is not supported for Operator.");
            },
            nb::is_operator())
        .def(
            "__isub__",
            [](Operator<Prec, Space>&, const Operator<Prec, Space>&) -> Operator<Prec, Space>& {
                throw nb::type_error("In-place subtraction (-=) is not supported for Operator.");
            },
            nb::is_operator())
        .def(
            "__imul__",
            [](Operator<Prec, Space>&, const Operator<Prec, Space>&) -> Operator<Prec, Space>& {
                throw nb::type_error("In-place multiplication (*=) is not supported for Operator.");
            },
            nb::is_operator())
        .def(
            "__iadd__",
            [](Operator<Prec, Space>&, const PauliOperator<Prec>&) -> Operator<Prec, Space>& {
                throw nb::type_error(
                    "In-place addition (+=) is not supported for Operator with PauliOperator.");
            },
            nb::is_operator())
        .def(
            "__isub__",
            [](Operator<Prec, Space>&, const PauliOperator<Prec>&) -> Operator<Prec, Space>& {
                throw nb::type_error(
                    "In-place subtraction (-=) is not supported for Operator with PauliOperator.");
            },
            nb::is_operator())
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
    ground_state.def_ro("eigenvalue", &Operator<Prec, Space>::GroundState::eigenvalue)
        .def_ro("state", &Operator<Prec, Space>::GroundState::state);
}
}  // namespace internal
#endif
}  // namespace scaluq
