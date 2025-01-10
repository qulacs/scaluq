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

    friend void to_json(Json& j, const StateVectorBatched<Fp>& states) {
        auto amplitudes = states.get_amplitudes();

        j = Json{{"n_qubits", states._n_qubits},
                 {"batch_size", states._batch_size},
                 {"batched_amplitudes", Json::array()}};
        for (std::uint32_t i = 0; i < amplitudes.size(); ++i) {
            Json state = {{"amplitudes", Json::array()}};
            for (const auto& amp : amplitudes[i]) {
                state["amplitudes"].push_back({{"real", amp.real()}, {"imag", amp.imag()}});
            }
            j["batched_amplitudes"].push_back(state);
        }
    }
    friend void from_json(const Json& j, StateVectorBatched<Fp>& states) {
        std::uint32_t b = j.at("batch_size").get<std::uint32_t>();
        std::uint32_t n = j.at("n_qubits").get<std::uint32_t>();
        states = StateVectorBatched(b, n);

        const auto& batched_amplitudes = j.at("batched_amplitudes");
        std::vector res(b, std::vector<Kokkos::complex<Fp>>(1ULL << n));
        for (std::uint32_t i = 0; i < b; ++i) {
            const auto& amplitudes = batched_amplitudes[i].at("amplitudes");
            for (std::uint32_t j = 0; j < (1ULL << n); ++j) {
                Fp real = amplitudes[j].at("real").get<Fp>();
                Fp imag = amplitudes[j].at("imag").get<Fp>();
                res[i][j] = Kokkos::complex<Fp>(real, imag);
            }
        }
        states.load(res);
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_state_state_vector_batched_hpp(nb::module_& m) {
    nb::class_<StateVectorBatched<Fp>>(
        m,
        "StateVectorBatched",
        DocString()
            .desc("Batched vector representation of quantum state.")
            .desc("Qubit index starts from 0. If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ "
                  "are $b_i$, the state is $\\sum_i b_i 2^i$.")
            .build_as_google_style()
            .c_str())
        // Constructor: batch size and number of qubits
        .def(nb::init<std::uint64_t, std::uint64_t>(),
             "batch_size"_a,
             "n_qubits"_a,
             DocString()
                 .desc("Construct batched state vector with specified batch size and qubits.")
                 .arg("batch_size", "int", "Number of batches.")
                 .arg("n_qubits", "int", "Number of qubits in each state vector.")
                 .ex(DocString::Code({">>> states = StateVectorBatched(3, 2)",
                                      ">>> print(states)",
                                      " *** Quantum States ***",
                                      " * Qubit Count : 2",
                                      " * Dimension : 4",
                                      "--------------------",
                                      " * Batch_id : 0",
                                      " * State vector : ",
                                      "  00 : (1,0)\n  01 : (0,0)\n  10 : (0,0)\n  11 : (0,0)",
                                      "--------------------",
                                      " * Batch_id : 1",
                                      " * State vector : ",
                                      "  00 : (1,0)\n  01 : (0,0)\n  10 : (0,0)\n  11 : (0,0)"}))
                 .build_as_google_style()
                 .c_str())
        // Constructor: Copy constructor
        .def(nb::init<const StateVectorBatched<Fp>&>(),
             "other"_a,
             DocString()
                 .desc("Construct a batched state vector by copying another batched state.")
                 .arg("other", "StateVectorBatched", "The batched state vector to copy.")
                 .build_as_google_style()
                 .c_str())
        // Basic getters
        .def("n_qubits",
             &StateVectorBatched<Fp>::n_qubits,
             DocString()
                 .desc("Get the number of qubits in each state vector.")
                 .ret("int", "The number of qubits.")
                 .build_as_google_style()
                 .c_str())
        .def("dim",
             &StateVectorBatched<Fp>::dim,
             DocString()
                 .desc("Get the dimension of each state vector (=$2^{\\mathrm{n\\_qubits}}$).")
                 .ret("int", "The dimension of the vector.")
                 .build_as_google_style()
                 .c_str())
        .def("batch_size",
             &StateVectorBatched<Fp>::batch_size,
             DocString()
                 .desc("Get the batch size (number of state vectors).")
                 .ret("int", "The batch size.")
                 .build_as_google_style()
                 .c_str())
        // State manipulation methods
        .def("set_state_vector",
             &StateVectorBatched<Fp>::set_state_vector,
             "state"_a,
             DocString()
                 .desc("Set all state vectors in the batch to the given state.")
                 .arg("state", "StateVector", "State to set for all batches.")
                 .build_as_google_style()
                 .c_str())
        .def("set_state_vector_at",
             &StateVectorBatched<Fp>::set_state_vector_at,
             "batch_id"_a,
             "state"_a,
             DocString()
                 .desc("Set the state vector at a specific batch index.")
                 .arg("batch_id", "int", "Index in batch to set.")
                 .arg("state", "StateVector", "State to set at the specified index.")
                 .build_as_google_style()
                 .c_str())
        .def("get_state_vector_at",
             &StateVectorBatched<Fp>::get_state_vector_at,
             "batch_id"_a,
             DocString()
                 .desc("Get the state vector at a specific batch index.")
                 .arg("batch_id", "int", "Index in batch to get.")
                 .ret("StateVector", "The state vector at the specified batch index.")
                 .build_as_google_style()
                 .c_str())
        // State initialization methods
        .def("set_zero_state",
             &StateVectorBatched<Fp>::set_zero_state,
             DocString().desc("Initialize all states to |0...0⟩.").build_as_google_style().c_str())
        .def("set_computational_basis",
             &StateVectorBatched<Fp>::set_computational_basis,
             "basis"_a,
             DocString()
                 .desc("Set all states to the specified computational basis state.")
                 .arg("basis", "int", "Index of the computational basis state.")
                 .build_as_google_style()
                 .c_str())
        .def("set_zero_norm_state",
             &StateVectorBatched<Fp>::set_zero_norm_state,
             DocString().desc("Set all amplitudes to zero.").build_as_google_style().c_str())
        // Haar random state methods
        .def(
            "set_Haar_random_state",
            [](StateVectorBatched<Fp>& states,
               std::uint64_t batch_size,
               std::uint64_t n_qubits,
               bool set_same_state,
               std::optional<std::uint64_t> seed) {
                states.set_Haar_random_state(
                    batch_size, n_qubits, set_same_state, seed.value_or(std::random_device()()));
            },
            "batch_size"_a,
            "n_qubits"_a,
            "set_same_state"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Initialize with Haar random states.")
                .arg("batch_size", "int", "Number of states in batch.")
                .arg("n_qubits", "int", "Number of qubits per state.")
                .arg(
                    "set_same_state", "bool", "Whether to set all states to the same random state.")
                .arg("seed", "int, optional", "Random seed (default: random).")
                .build_as_google_style()
                .c_str())
        .def_static(
            "Haar_random_state",
            [](std::uint64_t batch_size,
               std::uint64_t n_qubits,
               bool set_same_state,
               std::optional<std::uint64_t> seed) {
                return StateVectorBatched<Fp>::Haar_random_state(
                    batch_size, n_qubits, set_same_state, seed.value_or(std::random_device()()));
            },
            "batch_size"_a,
            "n_qubits"_a,
            "set_same_state"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Construct :class:`StateVectorBatched` with Haar random state.")
                .arg("batch_size", "int", "Number of states in batch.")
                .arg("n_qubits", "int", "Number of qubits per state.")
                .arg(
                    "set_same_state", "bool", "Whether to set all states to the same random state.")
                .arg("seed", "int, optional", "Random seed (default: random).")
                .ret("StateVectorBatched", "New batched state vector with random states.")
                .build_as_google_style()
                .c_str())
        // Measurement and probability methods
        .def("get_squared_norm",
             &StateVectorBatched<Fp>::get_squared_norm,
             DocString()
                 .desc("Get squared norm for each state in the batch.")
                 .ret("list[float]", "List of squared norms.")
                 .build_as_google_style()
                 .c_str())
        .def("normalize",
             &StateVectorBatched<Fp>::normalize,
             DocString().desc("Normalize all states in the batch.").build_as_google_style().c_str())
        .def("get_zero_probability",
             &StateVectorBatched<Fp>::get_zero_probability,
             "target_qubit_index"_a,
             DocString()
                 .desc("Get probability of measuring |0⟩ on specified qubit for each state.")
                 .arg("target_qubit_index", "int", "Index of qubit to measure.")
                 .ret("list[float]", "Probabilities for each state in batch.")
                 .build_as_google_style()
                 .c_str())
        .def("get_marginal_probability",
             &StateVectorBatched<Fp>::get_marginal_probability,
             "measured_values"_a,
             DocString()
                 .desc("Get marginal probabilities for specified measurement outcomes.")
                 .arg("measured_values", "list[int]", "Measurement configuration.")
                 .ret("list[float]", "Probabilities for each state in batch.")
                 .build_as_google_style()
                 .c_str())
        // Entropy and sampling methods
        .def("get_entropy",
             &StateVectorBatched<Fp>::get_entropy,
             DocString()
                 .desc("Calculate von Neumann entropy for each state.")
                 .ret("list[float]", "Entropy values for each state.")
                 .build_as_google_style()
                 .c_str())
        .def(
            "sampling",
            [](const StateVectorBatched<Fp>& states,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return states.sampling(sampling_count, seed.value_or(std::random_device()()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Sample from the probability distribution of each state.")
                .arg("sampling_count", "int", "Number of samples to take.")
                .arg("seed", "int, optional", "Random seed (default: random).")
                .ret("list[list[int]]", "Samples for each state in batch.")
                .build_as_google_style()
                .c_str())
        // State manipulation methods
        .def("add_state_vector_with_coef",
             &StateVectorBatched<Fp>::add_state_vector_with_coef,
             "coef"_a,
             "states"_a,
             DocString()
                 .desc("Add another batched state vector multiplied by a coefficient.")
                 .arg("coef", "complex", "Coefficient to multiply with states.")
                 .arg("states", "StateVectorBatched", "States to add.")
                 .build_as_google_style()
                 .c_str())
        .def("multiply_coef",
             &StateVectorBatched<Fp>::multiply_coef,
             "coef"_a,
             DocString()
                 .desc("Multiply all states by a coefficient.")
                 .arg("coef", "complex", "Coefficient to multiply.")
                 .build_as_google_style()
                 .c_str())
        // Data access methods
        .def("load",
             &StateVectorBatched<Fp>::load,
             "states"_a,
             DocString()
                 .desc("Load amplitudes for all states in batch.")
                 .arg("states", "list[list[complex]]", "Amplitudes for each state.")
                 .build_as_google_style()
                 .c_str())
        .def("get_amplitudes",
             &StateVectorBatched<Fp>::get_amplitudes,
             DocString()
                 .desc("Get amplitudes of all states in batch.")
                 .ret("list[list[complex]]", "Amplitudes for each state.")
                 .build_as_google_style()
                 .c_str())
        // Copy and string representation
        .def("copy",
             &StateVectorBatched<Fp>::copy,
             DocString()
                 .desc("Create a deep copy of this batched state vector.")
                 .ret("StateVectorBatched", "New copy of the states.")
                 .build_as_google_style()
                 .c_str())
        .def("to_string",
             &StateVectorBatched<Fp>::to_string,
             DocString()
                 .desc("Get string representation of the batched states.")
                 .ret("str", "String representation of states.")
                 .build_as_google_style()
                 .c_str())
        .def("__str__",
             &StateVectorBatched<Fp>::to_string,
             DocString()
                 .desc("Get string representation of the batched states.")
                 .ret("str", "String representation of states.")
                 .build_as_google_style()
                 .c_str())
        // JSON serialization
        .def(
            "to_json",
            [](const StateVectorBatched<Fp>& states) { return Json(states).dump(); },
            DocString()
                .desc("Convert states to JSON string.")
                .ret("str", "JSON representation of states.")
                .build_as_google_style()
                .c_str())
        .def(
            "load_json",
            [](StateVectorBatched<Fp>& states, const std::string& str) {
                states = nlohmann::json::parse(str);
            },
            "json_str"_a,
            DocString()
                .desc("Load states from JSON string.")
                .arg("json_str", "str", "JSON string to load from.")
                .build_as_google_style()
                .c_str());
}
}  // namespace internal
#endif

}  // namespace scaluq
