#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <std::floating_point Fp>
class StateVectorBatched {
    std::uint64_t _batch_size;
    std::uint64_t _n_qubits;
    std::uint64_t _dim;

    // static_assert(std::is_same_v<Space, HostSpace> || std::is_same_v<Space, DefaultSpace>,
    //               "Unsupported execution space tag");

public:
    Kokkos::View<Kokkos::complex<Fp>**, Kokkos::LayoutRight> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(std::uint64_t batch_size, std::uint64_t n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::uint64_t batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector<Fp>& state);

    void set_state_vector_at(std::uint64_t batch_id, const StateVector<Fp>& state);

    [[nodiscard]] StateVector<Fp> get_state_vector_at(std::uint64_t batch_id) const;

    void set_zero_state() { set_computational_basis(0); }

    void set_computational_basis(std::uint64_t basis);

    void set_zero_norm_state();

    void set_Haar_random_state(std::uint64_t batch_size,
                               std::uint64_t n_qubits,
                               bool set_same_state,
                               std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<std::uint64_t>> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    [[nodiscard]] static StateVectorBatched Haar_random_state(
        std::uint64_t batch_size,
        std::uint64_t n_qubits,
        bool set_same_state,
        std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<Kokkos::complex<Fp>>> get_amplitudes() const;

    [[nodiscard]] std::vector<Fp> get_squared_norm() const;

    void normalize();

    [[nodiscard]] std::vector<Fp> get_zero_probability(std::uint64_t target_qubit_index) const;

    [[nodiscard]] std::vector<Fp> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;

    [[nodiscard]] std::vector<Fp> get_entropy() const;

    void add_state_vector_with_coef(const Kokkos::complex<Fp>& coef,
                                    const StateVectorBatched& states);

    void multiply_coef(const Kokkos::complex<Fp>& coef);

    void load(const std::vector<std::vector<Kokkos::complex<Fp>>>& states);

    [[nodiscard]] StateVectorBatched copy() const;

    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states) {
        os << states.to_string();
        return os;
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_state_state_vector_batched_hpp(nb::module_& m) {
    nb::class_<StateVectorBatched<double>>(
        m,
        "StateVectorBatched",
        "Batched vector representation of quantum state.\n\n.. note:: Qubit index is start from 0. "
        "If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ is $b_i$, the state is $\\sum_i b_i "
        "2^i$.")
        .def(nb::init<std::uint64_t, std::uint64_t>(),
             "Construct batched state vector with specified batch size and qubits.")
        .def(nb::init<const StateVectorBatched<double>&>(),
             "Constructing batched state vector by copying other batched state.")
        .def("n_qubits", &StateVectorBatched<double>::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVectorBatched<double>::dim,
             "Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
        .def("batch_size", &StateVectorBatched<double>::batch_size, "Get batch size.")
        .def("set_state_vector",
             nb::overload_cast<const StateVector<double>&>(
                 &StateVectorBatched<double>::set_state_vector),
             "Set the state vector for all batches.")
        .def("set_state_vector_at",
             nb::overload_cast<std::uint64_t, const StateVector<double>&>(
                 &StateVectorBatched<double>::set_state_vector_at),
             "Set the state vector for a specific batch.")
        .def("get_state_vector_at",
             &StateVectorBatched<double>::get_state_vector_at,
             "Get the state vector for a specific batch.")
        .def("set_zero_state",
             &StateVectorBatched<double>::set_zero_state,
             "Initialize all batches with computational basis $\\ket{00\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVectorBatched<double>::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVectorBatched<double>::set_computational_basis,
             "Initialize with computational basis \\ket{\\mathrm{basis}}.")
        .def(
            "sampling",
            [](const StateVectorBatched<double>& states,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return states.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is `list[list[int]]` with the `sampling_count` "
            "length.")
        .def_static(
            "Haar_random_state",
            [](std::uint64_t batch_size,
               std::uint64_t n_qubits,
               bool set_same_state,
               std::optional<std::uint64_t> seed) {
                return StateVectorBatched<double>::Haar_random_state(
                    batch_size, n_qubits, set_same_state, seed.value_or(std::random_device{}()));
            },
            "batch_size"_a,
            "n_qubits"_a,
            "set_same_state"_a,
            "seed"_a = std::nullopt,
            "Construct batched state vectors with Haar random states. If seed is not "
            "specified, the value from random device is used.")
        .def("get_amplitudes",
             &StateVectorBatched<double>::get_amplitudes,
             "Get all amplitudes with as `list[list[complex]]`.")
        .def("get_squared_norm",
             &StateVectorBatched<double>::get_squared_norm,
             "Get squared norm of each state in the batch. $\\braket{\\psi|\\psi}$.")
        .def("normalize",
             &StateVectorBatched<double>::normalize,
             "Normalize each state in the batch (let $\\braket{\\psi|\\psi} = 1$ by "
             "multiplying coef).")
        .def("get_zero_probability",
             &StateVectorBatched<double>::get_zero_probability,
             "Get the probability to observe $\\ket{0}$ at specified index for each state in "
             "the batch.")
        .def("get_marginal_probability",
             &StateVectorBatched<double>::get_marginal_probability,
             "Get the marginal probability to observe as specified for each state in the batch. "
             "Specify the result as n-length list. `0` and `1` represent the qubit is observed "
             "and get the value. `2` represents the qubit is not observed.")
        .def("get_entropy",
             &StateVectorBatched<double>::get_entropy,
             "Get the entropy of each state in the batch.")
        .def("add_state_vector_with_coef",
             &StateVectorBatched<double>::add_state_vector_with_coef,
             "Add other batched state vectors with multiplying the coef and make superposition. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}"
             "\\ket{\\mathrm{states}}$.")
        .def("load",
             &StateVectorBatched<double>::load,
             "Load batched amplitudes from `list[list[complex]]`.")
        .def(
            "copy", &StateVectorBatched<double>::copy, "Create a copy of the batched state vector.")
        .def("to_string", &StateVectorBatched<double>::to_string, "Information as `str`.")
        .def("__str__", &StateVectorBatched<double>::to_string, "Information as `str`.");
}
}  // namespace internal
#endif

}  // namespace scaluq
