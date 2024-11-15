#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace scaluq {

using HostSpace = Kokkos::HostSpace;
using DefaultSpace = Kokkos::DefaultExecutionSpace;

template <std::floating_point Fp>
class StateVector {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    using ComplexType = Kokkos::complex<Fp>;

public:
    static constexpr std::uint64_t UNMEASURED = 2;
    Kokkos::View<ComplexType*> _raw;
    StateVector() = default;
    StateVector(std::uint64_t n_qubits);
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at(std::uint64_t index, ComplexType c);

    /**
     * @attention Very slow. You should use get_amplitudes() instead if you can.
     */
    [[nodiscard]] ComplexType get_amplitude_at(std::uint64_t index);

    [[nodiscard]] static StateVector Haar_random_state(std::uint64_t n_qubits,
                                                       std::uint64_t seed = std::random_device()());

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);
    void set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::vector<ComplexType> get_amplitudes() const;

    [[nodiscard]] Fp get_squared_norm() const;

    void normalize();

    [[nodiscard]] Fp get_zero_probability(std::uint64_t target_qubit_index) const;

    [[nodiscard]] Fp get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] Fp get_entropy() const;

    void add_state_vector_with_coef(ComplexType coef, const StateVector& state);
    void multiply_coef(ComplexType coef);

    [[nodiscard]] std::vector<std::uint64_t> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    void load(const std::vector<ComplexType>& other);

    [[nodiscard]] StateVector copy() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state) {
        os << state.to_string();
        return os;
    }

    [[nodiscard]] std::string to_string() const;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_state_state_vector_hpp(nb::module_& m) {
    nb::class_<StateVector<double>>(
        m,
        "StateVector_double",
        "Vector representation of quantum state.\n\n.. note:: Qubit index is "
        "start from 0. If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ is "
        "$b_i$, the state is $\\sum_i b_i 2^i$.")
        .def(nb::init<std::uint64_t>(),
             "Construct state vector with specified qubits, initialized with computational "
             "basis $\\ket{0\\dots0}$.")
        .def(nb::init<const StateVector<double>&>(),
             "Constructing state vector by copying other state.")
        .def_static(
            "Haar_random_state",
            [](std::uint64_t n_qubits, std::optional<std::uint64_t> seed) {
                return StateVector<double>::Haar_random_state(
                    n_qubits, seed.value_or(std::random_device{}()));
            },
            "n_qubits"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Construct :class:`StateVector` with Haar random state.")
                .arg("n_qubits", "int", "number of qubits")
                .arg("seed",
                     "int | None",
                     true,
                     "random seed",
                     "If not specified, the value from random device is used.")
                .ex(DocString::Code(
                    {">>> state = StateVector.Haar_random_state(2)",
                     ">>> print(state.get_amplitudes())",
                     "[(-0.3188299516496241+0.6723250989136779j), "
                     "(-0.253461343768224-0.22430415678425403j), "
                     "(0.24998142919420457+0.33096908710840045j), "
                     "(0.2991187916479724+0.2650813322096342j)]",
                     ">>> print(StateVector.Haar_random_state(2).get_amplitudes()) # If seed is "
                     "not specified, generated vector differs.",
                     "[(-0.49336775961196616-0.3319437726884906j), "
                     "(-0.36069529482031787+0.31413708595210815j), "
                     "(-0.3654176892043237-0.10307602590749808j), "
                     "(-0.18175679804035652+0.49033467421609994j)]",
                     ">>> print(StateVector.Haar_random_state(2, 0).get_amplitudes())",
                     "[(0.030776817573663098-0.7321137912473642j), "
                     "(0.5679070655936114-0.14551095055034327j), "
                     "(-0.0932995615041323-0.07123201881040941j), "
                     "(0.15213024630399696-0.2871374092016799j)]",
                     ">>> print(StateVector.Haar_random_state(2, 0).get_amplitudes()) # If same "
                     "seed is specified, same vector is generated.",
                     "[(0.030776817573663098-0.7321137912473642j), "
                     "(0.5679070655936114-0.14551095055034327j), "
                     "(-0.0932995615041323-0.07123201881040941j), "
                     "(0.15213024630399696-0.2871374092016799j)]"}))
                .build_as_google_style()
                .c_str())
        .def("set_amplitude_at",
             &StateVector<double>::set_amplitude_at,
             "Manually set amplitude at one index.")
        .def("get_amplitude_at",
             &StateVector<double>::get_amplitude_at,
             "Get amplitude at one index.\n\n.. note:: If you want to get all amplitudes, you "
             "should "
             "use `StateVector::get_amplitudes()`.")
        .def("set_zero_state",
             &StateVector<double>::set_zero_state,
             "Initialize with computational basis $\\ket{00\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVector<double>::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVector<double>::set_computational_basis,
             "Initialize with computational basis \\ket{\\mathrm{basis}}.")
        .def("get_amplitudes",
             &StateVector<double>::get_amplitudes,
             "Get all amplitudes with as `list[complex]`.")
        .def("n_qubits", &StateVector<double>::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVector<double>::dim,
             "Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
        .def("get_squared_norm",
             &StateVector<double>::get_squared_norm,
             "Get squared norm of the state. $\\braket{\\psi|\\psi}$.")
        .def("normalize",
             &StateVector<double>::normalize,
             "Normalize state (let $\\braket{\\psi|\\psi} = 1$ by multiplying coef).")
        .def("get_zero_probability",
             &StateVector<double>::get_zero_probability,
             "Get the probability to observe $\\ket{0}$ at specified index.")
        .def("get_marginal_probability",
             &StateVector<double>::get_marginal_probability,
             "Get the marginal probability to observe as specified. Specify the result as n-length "
             "list. `0` and `1` represent the qubit is observed and get the value. `2` represents "
             "the qubit is not observed.")
        .def("get_entropy", &StateVector<double>::get_entropy, "Get the entropy of the vector.")
        .def("add_state_vector_with_coef",
             &StateVector<double>::add_state_vector_with_coef,
             "add other state vector with multiplying the coef and make superposition. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}"
             "\\ket{\\mathrm{"
             "state}}$.")
        .def("multiply_coef",
             &StateVector<double>::multiply_coef,
             "Multiply coef. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\mathrm{coef}\\ket{\\mathrm{this}}$.")
        .def(
            "sampling",
            [](const StateVector<double>& state,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is `list[int]` with the `sampling_count` length.")
        .def("to_string", &StateVector<double>::to_string, "Information as `str`.")
        .def(
            "load", &StateVector<double>::load, "Load amplitudes of `list[int]` with `dim` length.")
        .def("__str__", &StateVector<double>::to_string, "Information as `str`.")
        .def_ro_static("UNMEASURED",
                       &StateVector<double>::UNMEASURED,
                       "Constant used for `StateVector::get_marginal_probability` to express the "
                       "the qubit is not measured.");
}
}  // namespace internal
#endif
}  // namespace scaluq
