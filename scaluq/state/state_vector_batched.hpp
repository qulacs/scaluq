#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {
class StateVectorBatched {
    std::uint64_t _batch_size;
    std::uint64_t _n_qubits;
    std::uint64_t _dim;

public:
    Kokkos::View<Complex**, Kokkos::LayoutRight> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(std::uint64_t batch_size, std::uint64_t n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::uint64_t batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector& state);
    void set_state_vector(std::uint64_t batch_id, const StateVector& state);
    [[nodiscard]] StateVector get_state_vector_at(std::uint64_t batch_id) const;

    void set_zero_state() const;
    void set_computational_basis(std::uint64_t basis) const;
    void set_zero_norm_state() const;

    [[nodiscard]] std::vector<std::vector<std::uint64_t>> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    [[nodiscard]] static StateVectorBatched Haar_random_states(
        std::uint64_t batch_size,
        std::uint64_t n_qubits,
        bool set_same_state,
        std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<Complex>> get_amplitudes() const;

    [[nodiscard]] std::vector<double> get_squared_norm() const;

    void normalize();

    [[nodiscard]] std::vector<double> get_zero_probability(std::uint64_t target_qubit_index) const;
    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] std::vector<double> get_entropy() const;

    void add_state_vector(const StateVectorBatched& states);
    void add_state_vector_with_coef(const Complex& coef, const StateVectorBatched& states);
    void multiply_coef(const Complex& coef);

    void load(const std::vector<std::vector<Complex>>& states) const;
    [[nodiscard]] StateVectorBatched copy() const;

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states);
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_state_state_vector_batched_hpp(nb::module_& m) {
    nb::class_<StateVectorBatched>(
        m,
        "StateVectorBatched",
        "Batched vector representation of quantum state.\n\n.. note:: Qubit index is start from 0. "
        "If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ is $b_i$, the state is $\\sum_i b_i "
        "2^i$.")
        .def(nb::init<std::uint64_t, std::uint64_t>(),
             "Construct batched state vector with specified batch size and qubits.")
        .def(nb::init<const StateVectorBatched&>(),
             "Constructing batched state vector by copying other batched state.")
        .def("n_qubits", &StateVectorBatched::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVectorBatched::dim,
             "Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
        .def("batch_size", &StateVectorBatched::batch_size, "Get batch size.")
        .def("set_state_vector",
             nb::overload_cast<const StateVector&>(&StateVectorBatched::set_state_vector),
             "Set the state vector for all batches.")
        .def("set_state_vector",
             nb::overload_cast<std::uint64_t, const StateVector&>(
                 &StateVectorBatched::set_state_vector),
             "Set the state vector for a specific batch.")
        .def("get_state_vector_at",
             &StateVectorBatched::get_state_vector_at,
             "Get the state vector for a specific batch.")
        .def("set_zero_state",
             &StateVectorBatched::set_zero_state,
             "Initialize all batches with computational basis $\\ket{00\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVectorBatched::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVectorBatched::set_computational_basis,
             "Initialize with computational basis \\ket{\\mathrm{basis}}.")
        .def(
            "sampling",
            [](const StateVectorBatched& states,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return states.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            "Sampling specified times. Result is `list[list[int]]` with the `sampling_count` "
            "length.")
        .def_static(
            "Haar_random_states",
            [](std::uint64_t batch_size,
               std::uint64_t n_qubits,
               bool set_same_state,
               std::optional<std::uint64_t> seed) {
                return StateVectorBatched::Haar_random_states(
                    batch_size, n_qubits, set_same_state, seed.value_or(std::random_device{}()));
            },
            "batch_size"_a,
            "n_qubits"_a,
            "set_same_state"_a,
            "seed"_a = std::nullopt,
            "Construct batched state vectors with Haar random states. If seed is not "
            "specified, the value from random device is used.")
        .def("amplitudes",
             &StateVectorBatched::get_amplitudes,
             "Get all amplitudes with as `list[list[complex]]`.")
        .def("get_squared_norm",
             &StateVectorBatched::get_squared_norm,
             "Get squared norm of each state in the batch. $\\braket{\\psi|\\psi}$.")
        .def("normalize",
             &StateVectorBatched::normalize,
             "Normalize each state in the batch (let $\\braket{\\psi|\\psi} = 1$ by "
             "multiplying coef).")
        .def("get_zero_probability",
             &StateVectorBatched::get_zero_probability,
             "Get the probability to observe $\\ket{0}$ at specified index for each state in "
             "the batch.")
        .def("get_marginal_probability",
             &StateVectorBatched::get_marginal_probability,
             "Get the marginal probability to observe as specified for each state in the batch. "
             "Specify the result as n-length list. `0` and `1` represent the qubit is observed "
             "and get the value. `2` represents the qubit is not observed.")
        .def("get_entropy",
             &StateVectorBatched::get_entropy,
             "Get the entropy of each state in the batch.")
        .def("add_state_vector",
             &StateVectorBatched::add_state_vector,
             "Add other batched state vectors and make superposition. $\\ket{\\mathrm{this}} "
             "\\leftarrow \\ket{\\mathrm{this}} + \\ket{\\mathrm{states}}$.")
        .def("add_state_vector_with_coef",
             &StateVectorBatched::add_state_vector_with_coef,
             "Add other batched state vectors with multiplying the coef and make superposition. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}"
             "\\ket{\\mathrm{states}}$.")
        .def("load",
             &StateVectorBatched::load,
             "Load batched amplitudes from `list[list[complex]]`.")
        .def("copy", &StateVectorBatched::copy, "Create a copy of the batched state vector.")
        .def("to_string", &StateVectorBatched::to_string, "Information as `str`.")
        .def("__str__", &StateVectorBatched::to_string, "Information as `str`.");
}
}  // namespace internal
#endif

}  // namespace scaluq
