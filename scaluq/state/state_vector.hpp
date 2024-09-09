#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "../types.hpp"

namespace scaluq {
class StateVector {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;

public:
    static constexpr std::uint64_t UNMEASURED = 2;
    Kokkos::View<Complex*> _raw;
    StateVector() = default;
    StateVector(std::uint64_t n_qubits);
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at(std::uint64_t index, const Complex& c);

    /**
     * @attention Very slow. You should use get_amplitudes() instead if you can.
     */
    [[nodiscard]] Complex get_amplitude_at(std::uint64_t index) const;

    [[nodiscard]] static StateVector Haar_random_state(std::uint64_t n_qubits,
                                                       std::uint64_t seed = std::random_device()());

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);

    [[nodiscard]] std::uint64_t n_qubits() const;

    [[nodiscard]] std::uint64_t dim() const;

    [[nodiscard]] std::vector<Complex> get_amplitudes() const;

    [[nodiscard]] double get_squared_norm() const;

    void normalize();

    [[nodiscard]] double get_zero_probability(std::uint64_t target_qubit_index) const;
    [[nodiscard]] double get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] double get_entropy() const;

    void add_state_vector(const StateVector& state);
    void add_state_vector_with_coef(const Complex& coef, const StateVector& state);
    void multiply_coef(const Complex& coef);

    [[nodiscard]] std::vector<std::uint64_t> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    void load(const std::vector<Complex>& other);

    [[nodiscard]] StateVector copy() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state);

    [[nodiscard]] std::string to_string() const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_state_state_vector_hpp(nb::module_& m) {
    nb::class_<StateVector>(m,
                            "StateVector",
                            DocString()
                                .desc("Vector representation of quantum state.")
                                .desc("Qubit index is "
                                      "start from 0. If the i-th value of the vector is "
                                      "$a_i$, the state is $\\sum_i a_i \\ket{i}$.")
                                .build_as_google_style()
                                .c_str())
        .def(nb::init<std::uint64_t>(),
             DocString()
                 .desc(
                     "Construct state vector with specified qubits, initialized with computational "
                     "basis $\\ket{0\\dots0}$.")
                 .build_as_google_style()
                 .c_str())
        .def(nb::init<const StateVector&>(), "Constructing state vector by copying other state.")
        .def_static(
            "Haar_random_state",
            [](std::uint64_t n_qubits, std::optional<std::uint64_t> seed) {
                return StateVector::Haar_random_state(n_qubits,
                                                      seed.value_or(std::random_device{}()));
            },
            "n_qubits"_a,
            "seed"_a = std::nullopt,
            "Constructing state vector with Haar random state. If seed is not specified, the value "
            "from random device is used.")
        .def("set_amplitude_at",
             &StateVector::set_amplitude_at,
             "Manually set amplitude at one index.")
        .def("get_amplitude_at",
             &StateVector::get_amplitude_at,
             "index"_a,
             DocString()
                 .desc("Get amplitude at one index.")
                 .arg({"index",
                       "int",
                       false,
                       {"Index of state vector.",
                        "This is read as binary. k-th bit of index represents k-th qubit."}})
                 .ret({"complex", {"Amplitude at specified index"}})
                 .ex(DocString::Code{">>> state = StateVector(2)",
                                     ">>> state.load([1+2j, 3+4j, 5+6j, 7+8j])",
                                     ">>> state.get_amplitude_at(0)",
                                     "(1+2j)",
                                     ">>> state.get_amplitude_at(1)",
                                     "(3+4j)",
                                     ">>> state.get_amplitude_at(2)",
                                     "(5+6j)",
                                     ">>> state.get_amplitude_at(3)",
                                     "(7+8j)"})
                 .note("If you want to get amplitudes at all indices, you should use "
                       ":meth:`.get_amplitudes`.")
                 .build_as_google_style()
                 .c_str())
        .def("set_zero_state",
             &StateVector::set_zero_state,
             "Initialize with computational basis $\\ket{00\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVector::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVector::set_computational_basis,
             "Initialize with computational basis \\ket{\\mathrm{basis}}.")
        .def("get_amplitudes",
             &StateVector::get_amplitudes,
             "Get all amplitudes with as `list[complex]`.")
        .def("n_qubits", &StateVector::n_qubits, "Get num of qubits.")
        .def("dim", &StateVector::dim, "Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
        .def("get_squared_norm",
             &StateVector::get_squared_norm,
             "Get squared norm of the state. $\\braket{\\psi|\\psi}$.")
        .def("normalize",
             &StateVector::normalize,
             "Normalize state (let $\\braket{\\psi|\\psi} = 1$ by multiplying coef).")
        .def("get_zero_probability",
             &StateVector::get_zero_probability,
             "Get the probability to observe $\\ket{0}$ at specified index.")
        .def("get_marginal_probability",
             &StateVector::get_marginal_probability,
             "Get the marginal probability to observe as specified. Specify the result as n-length "
             "list. `0` and `1` represent the qubit is observed and get the value. `2` represents "
             "the qubit is not observed.")
        .def("get_entropy", &StateVector::get_entropy, "Get the entropy of the vector.")
        .def("add_state_vector",
             &StateVector::add_state_vector,
             "Add other state vector and make superposition. $\\ket{\\mathrm{this}} "
             "\\leftarrow "
             "\\ket{\\mathrm{this}} + \\ket{\\mathrm{state}}$.")
        .def("add_state_vector_with_coef",
             &StateVector::add_state_vector_with_coef,
             "add other state vector with multiplying the coef and make superposition. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}"
             "\\ket{\\mathrm{"
             "state}}$.")
        .def("multiply_coef",
             &StateVector::multiply_coef,
             "Multiply coef. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\mathrm{coef}\\ket{\\mathrm{this}}$.")
        .def(
            "sampling",
            [](const StateVector& state,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is `list[int]` with the `sampling_count` length.")
        .def("to_string", &StateVector::to_string, "Information as `str`.")
        .def("load", &StateVector::load, "Load amplitudes of `list[int]` with `dim` length.")
        .def("__str__", &StateVector::to_string, "Information as `str`.")
        .def_ro_static("UNMEASURED",
                       &StateVector::UNMEASURED,
                       "Constant used for `StateVector::get_marginal_probability` to express the "
                       "the qubit is not measured.");
}
}  // namespace internal
#endif
}  // namespace scaluq
