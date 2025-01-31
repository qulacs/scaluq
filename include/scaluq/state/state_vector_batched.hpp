#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class StateVectorBatched {
    std::uint64_t _batch_size;
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    using FloatType = internal::Float<Prec>;
    using ComplexType = internal::Complex<Prec>;

    // static_assert(std::is_same_v<Space, HostSpace> || std::is_same_v<Space, DefaultSpace>,
    //               "Unsupported execution space tag");

public:
    Kokkos::View<ComplexType**, Kokkos::LayoutRight, Space> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(std::uint64_t batch_size, std::uint64_t n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::uint64_t batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector<Prec, Space>& state);

    void set_state_vector_at(std::uint64_t batch_id, const StateVector<Prec, Space>& state);

    [[nodiscard]] StateVector<Prec, Space> get_state_vector_at(std::uint64_t batch_id) const;

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

    [[nodiscard]] std::vector<std::vector<StdComplex>> get_amplitudes() const;

    [[nodiscard]] std::vector<double> get_squared_norm() const;

    void normalize();

    [[nodiscard]] std::vector<double> get_zero_probability(std::uint64_t target_qubit_index) const;

    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;

    [[nodiscard]] std::vector<double> get_entropy() const;

    void add_state_vector_with_coef(StdComplex coef, const StateVectorBatched& states);

    void multiply_coef(const StdComplex& coef);

    void load(const std::vector<std::vector<StdComplex>>& states);

    [[nodiscard]] StateVectorBatched copy() const;

    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states) {
        os << states.to_string();
        return os;
    }

    friend void to_json(Json& j, const StateVectorBatched<Prec, Space>& states) {
        j = Json{{"n_qubits", states._n_qubits},
                 {"batch_size", states._batch_size},
                 {"amplitudes", states.get_amplitudes()}};
    }
    friend void from_json(const Json& j, StateVectorBatched<Prec, Space>& states) {
        std::uint64_t b = j.at("batch_size").get<std::uint64_t>();
        std::uint64_t n = j.at("n_qubits").get<std::uint64_t>();
        states = StateVectorBatched(b, n);
        states.load(j.at("amplitudes").get<std::vector<std::vector<StdComplex>>>());

        // const auto& batched_amplitudes = j.at("batched_amplitudes");
        // std::vector res(b, std::vector<StdComplex>(1ULL << n));
        // for (std::uint32_t i = 0; i < b; ++i) {
        //     const auto& amplitudes = batched_amplitudes[i].at("amplitudes");
        //     for (std::uint32_t j = 0; j < (1ULL << n); ++j) {
        //         double real = amplitudes[j].at("real").get<double>();
        //         double imag = amplitudes[j].at("imag").get<double>();
        //         res[i][j] = ComplexType(real, imag);
        //     }
        // }
        // states.load(res);
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_state_state_vector_batched_hpp(nb::module_& m) {
    nb::class_<StateVectorBatched<Prec, Space>>(
        m,
        "StateVectorBatched",
        "Batched vector representation of quantum state.\n\n.. note:: Qubit index is start from 0. "
        "If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ is $b_i$, the state is $\\sum_i b_i "
        "2^i$.")
        .def(nb::init<std::uint64_t, std::uint64_t>(),
             "Construct batched state vector with specified batch size and qubits.")
        .def(nb::init<const StateVectorBatched<Prec, Space>&>(),
             "Constructing batched state vector by copying other batched state.")
        .def("n_qubits", &StateVectorBatched<Prec>::n_qubits, "Get num of qubits.")
        .def("dim",
             &StateVectorBatched<Prec, Space>::dim,
             "Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
        .def("batch_size", &StateVectorBatched<Prec, Space>::batch_size, "Get batch size.")
        .def("set_state_vector",
             nb::overload_cast<const StateVector<Prec, Space>&>(
                 &StateVectorBatched<Prec, Space>::set_state_vector),
             "Set the state vector for all batches.")
        .def("set_state_vector_at",
             nb::overload_cast<std::uint64_t, const StateVector<Prec, Space>&>(
                 &StateVectorBatched<Prec, Space>::set_state_vector_at),
             "Set the state vector for a specific batch.")
        .def("get_state_vector_at",
             &StateVectorBatched<Prec, Space>::get_state_vector_at,
             "Get the state vector for a specific batch.")
        .def("set_zero_state",
             &StateVectorBatched<Prec, Space>::set_zero_state,
             "Initialize all batches with computational basis $\\ket{00\\dots0}$.")
        .def("set_zero_norm_state",
             &StateVectorBatched<Prec, Space>::set_zero_norm_state,
             "Initialize with 0 (null vector).")
        .def("set_computational_basis",
             &StateVectorBatched<Prec, Space>::set_computational_basis,
             "Initialize with computational basis \\ket{\\mathrm{basis}}.")
        .def(
            "sampling",
            [](const StateVectorBatched<Prec, Space>& states,
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
                return StateVectorBatched<Prec, Space>::Haar_random_state(
                    batch_size, n_qubits, set_same_state, seed.value_or(std::random_device{}()));
            },
            "batch_size"_a,
            "n_qubits"_a,
            "set_same_state"_a,
            "seed"_a = std::nullopt,
            "Construct batched state vectors with Haar random states. If seed is not "
            "specified, the value from random device is used.")
        .def("get_amplitudes",
             &StateVectorBatched<Prec, Space>::get_amplitudes,
             "Get all amplitudes with as `list[list[complex]]`.")
        .def("get_squared_norm",
             &StateVectorBatched<Prec, Space>::get_squared_norm,
             "Get squared norm of each state in the batch. $\\braket{\\psi|\\psi}$.")
        .def("normalize",
             &StateVectorBatched<Prec, Space>::normalize,
             "Normalize each state in the batch (let $\\braket{\\psi|\\psi} = 1$ by "
             "multiplying coef).")
        .def("get_zero_probability",
             &StateVectorBatched<Prec, Space>::get_zero_probability,
             "Get the probability to observe $\\ket{0}$ at specified index for each state in "
             "the batch.")
        .def("get_marginal_probability",
             &StateVectorBatched<Prec, Space>::get_marginal_probability,
             "Get the marginal probability to observe as specified for each state in the batch. "
             "Specify the result as n-length list. `0` and `1` represent the qubit is observed "
             "and get the value. `2` represents the qubit is not observed.")
        .def("get_entropy",
             &StateVectorBatched<Prec, Space>::get_entropy,
             "Get the entropy of each state in the batch.")
        .def("add_state_vector_with_coef",
             &StateVectorBatched<Prec, Space>::add_state_vector_with_coef,
             "Add other batched state vectors with multiplying the coef and make superposition. "
             "$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef}"
             "\\ket{\\mathrm{states}}$.")
        .def("load",
             &StateVectorBatched<Prec, Space>::load,
             "Load batched amplitudes from `list[list[complex]]`.")
        .def("copy", &StateVectorBatched<Prec, Space>::copy, "Create a copy of the batched state vector.")
        .def("to_string", &StateVectorBatched<Prec, Space>::to_string, "Information as `str`.")
        .def("__str__", &StateVectorBatched<Prec, Space>::to_string, "Information as `str`.")
        .def(
            "to_json",
            [](const StateVectorBatched<Prec, Space>& states) { return Json(states).dump(); },
            "Get JSON representation of the states.")
        .def(
            "load_json",
            [](StateVectorBatched<Prec, Space>& states, const std::string& str) {
                states = nlohmann::json::parse(str);
            },
            "Read an object from the JSON representation of the states.");
}
}  // namespace internal
#endif

}  // namespace scaluq
