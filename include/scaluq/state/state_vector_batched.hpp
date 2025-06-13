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
    using ExecutionSpaceType = internal::SpaceType<Space>;

public:
    Kokkos::View<ComplexType**, Kokkos::LayoutRight, ExecutionSpaceType> _raw;
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

    void set_Haar_random_state(bool set_same_state, std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<std::uint64_t>> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    [[nodiscard]] static StateVectorBatched Haar_random_state(
        std::uint64_t batch_size,
        std::uint64_t n_qubits,
        bool set_same_state,
        std::uint64_t seed = std::random_device()());
    [[nodiscard]] static StateVectorBatched uninitialized_state(std::uint64_t batch_size,
                                                                std::uint64_t n_qubits);

    [[nodiscard]] std::vector<std::vector<StdComplex>> get_amplitudes() const;

    [[nodiscard]] std::vector<double> get_squared_norm() const;

    void normalize();

    [[nodiscard]] std::vector<double> get_zero_probability(std::uint64_t target_qubit_index) const;

    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;

    [[nodiscard]] std::vector<double> get_entropy() const;

    void add_state_vector_with_coef(const StdComplex& coef, const StateVectorBatched& states);

    void multiply_coef(const StdComplex& coef);

    void load(const std::vector<std::vector<StdComplex>>& states);

    [[nodiscard]] StateVectorBatched copy() const;

    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states) {
        os << states.to_string();
        return os;
    }

    friend void to_json(Json& j, const StateVectorBatched& states) {
        j = Json{{"n_qubits", states._n_qubits},
                 {"batch_size", states._batch_size},
                 {"amplitudes", states.get_amplitudes()}};
    }
    friend void from_json(const Json& j, StateVectorBatched& states) {
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
template <Precision Prec, ExecutionSpace Space>
void bind_state_state_vector_batched_hpp(nb::module_& m) {
    nb::class_<StateVectorBatched<Prec, Space>>(
        m,
        "StateVectorBatched",
        DocString()
            .desc("Batched vector representation of quantum state.")
            .desc("Qubit index starts from 0. If the amplitudes of $\\ket{b_{n-1}\\dots b_0}$ "
                  "are $b_i$, the state is $\\sum_i b_i 2^i$.")
            .desc("Given `batch_size: int, n_qubits: int`, construct a batched state vector with "
                  "specified batch size and qubits.")
            .desc("Given `other: StateVectorBatched`, Construct a batched state vector by copying "
                  "another batched state.")
            .ex(DocString::Code({">>> states = StateVectorBatched(3, 2)",
                                 ">>> print(states)",
                                 "Qubit Count : 2",
                                 "Dimension : 4",
                                 "--------------------",
                                 "Batch_id : 0",
                                 "State vector : ",
                                 "  00 : (1,0)",
                                 "  01 : (0,0)",
                                 "  10 : (0,0)",
                                 "  11 : (0,0)",
                                 "--------------------",
                                 "Batch_id : 1",
                                 "State vector : ",
                                 "  00 : (1,0)",
                                 "  01 : (0,0)",
                                 "  10 : (0,0)",
                                 "  11 : (0,0)",
                                 "--------------------",
                                 "Batch_id : 2",
                                 "State vector : ",
                                 "  00 : (1,0)",
                                 "  01 : (0,0)",
                                 "  10 : (0,0)",
                                 "  11 : (0,0)"}))
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
                                      "Qubit Count : 2",
                                      "Dimension : 4",
                                      "--------------------",
                                      "Batch_id : 0",
                                      "State vector : ",
                                      "  00 : (1,0)",
                                      "  01 : (0,0)",
                                      "  10 : (0,0)",
                                      "  11 : (0,0)",
                                      "--------------------",
                                      "Batch_id : 1",
                                      "State vector : ",
                                      "  00 : (1,0)",
                                      "  01 : (0,0)",
                                      "  10 : (0,0)",
                                      "  11 : (0,0)",
                                      "--------------------",
                                      "Batch_id : 2",
                                      "State vector : ",
                                      "  00 : (1,0)",
                                      "  01 : (0,0)",
                                      "  10 : (0,0)",
                                      "  11 : (0,0)"}))
                 .build_as_google_style()
                 .c_str())
        // Constructor: Copy constructor
        .def(nb::init<const StateVectorBatched<Prec, Space>&>(),
             "other"_a,
             DocString()
                 .desc("Construct a batched state vector by copying another batched state.")
                 .arg("other", "StateVectorBatched", "The batched state vector to copy.")
                 .build_as_google_style()
                 .c_str())
        // Basic getters
        .def("n_qubits",
             &StateVectorBatched<Prec, Space>::n_qubits,
             DocString()
                 .desc("Get the number of qubits in each state vector.")
                 .ret("int", "The number of qubits.")
                 .build_as_google_style()
                 .c_str())
        .def("dim",
             &StateVectorBatched<Prec, Space>::dim,
             DocString()
                 .desc("Get the dimension of each state vector (=$2^{\\mathrm{n\\_qubits}}$).")
                 .ret("int", "The dimension of the vector.")
                 .build_as_google_style()
                 .c_str())
        .def("batch_size",
             &StateVectorBatched<Prec, Space>::batch_size,
             DocString()
                 .desc("Get the batch size (number of state vectors).")
                 .ret("int", "The batch size.")
                 .build_as_google_style()
                 .c_str())
        // State manipulation methods
        .def("set_state_vector",
             &StateVectorBatched<Prec, Space>::set_state_vector,
             "state"_a,
             DocString()
                 .desc("Set all state vectors in the batch to the given state.")
                 .arg("state", "StateVector", "State to set for all batches.")
                 .build_as_google_style()
                 .c_str())
        .def("set_state_vector_at",
             &StateVectorBatched<Prec, Space>::set_state_vector_at,
             "batch_id"_a,
             "state"_a,
             DocString()
                 .desc("Set the state vector at a specific batch index.")
                 .arg("batch_id", "int", "Index in batch to set.")
                 .arg("state", "StateVector", "State to set at the specified index.")
                 .build_as_google_style()
                 .c_str())
        .def("get_state_vector_at",
             &StateVectorBatched<Prec, Space>::get_state_vector_at,
             "batch_id"_a,
             DocString()
                 .desc("Get the state vector at a specific batch index.")
                 .arg("batch_id", "int", "Index in batch to get.")
                 .ret("StateVector", "The state vector at the specified batch index.")
                 .build_as_google_style()
                 .c_str())
        // State initialization methods
        .def("set_zero_state",
             &StateVectorBatched<Prec, Space>::set_zero_state,
             DocString().desc("Initialize all states to |0...0⟩.").build_as_google_style().c_str())
        .def("set_computational_basis",
             &StateVectorBatched<Prec, Space>::set_computational_basis,
             "basis"_a,
             DocString()
                 .desc("Set all states to the specified computational basis state.")
                 .arg("basis", "int", "Index of the computational basis state.")
                 .build_as_google_style()
                 .c_str())
        .def("set_zero_norm_state",
             &StateVectorBatched<Prec, Space>::set_zero_norm_state,
             DocString().desc("Set all amplitudes to zero.").build_as_google_style().c_str())
        // Haar random state methods
        .def(
            "set_Haar_random_state",
            [](StateVectorBatched<Prec, Space>& states,
               bool set_same_state,
               std::optional<std::uint64_t> seed) {
                states.set_Haar_random_state(set_same_state, seed.value_or(std::random_device()()));
            },
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
                return StateVectorBatched<Prec, Space>::Haar_random_state(
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
        .def_static("uninitialized_state",
                    &StateVectorBatched<Prec, Space>::uninitialized_state,
                    "batch_size"_a,
                    "n_qubits"_a,
                    DocString()
                        .desc("Construct :class:`StateVectorBatched` without initializing.")
                        .arg("batch_size", "int", "Number of states in batch.")
                        .arg("n_qubits", "int", "number of qubits")
                        .build_as_google_style()
                        .c_str())
        // Measurement and probability methods
        .def("get_squared_norm",
             &StateVectorBatched<Prec, Space>::get_squared_norm,
             DocString()
                 .desc("Get squared norm for each state in the batch.")
                 .ret("list[float]", "List of squared norms.")
                 .build_as_google_style()
                 .c_str())
        .def("normalize",
             &StateVectorBatched<Prec, Space>::normalize,
             DocString().desc("Normalize all states in the batch.").build_as_google_style().c_str())
        .def("get_zero_probability",
             &StateVectorBatched<Prec, Space>::get_zero_probability,
             "target_qubit_index"_a,
             DocString()
                 .desc("Get probability of measuring |0⟩ on specified qubit for each state.")
                 .arg("target_qubit_index", "int", "Index of qubit to measure.")
                 .ret("list[float]", "Probabilities for each state in batch.")
                 .build_as_google_style()
                 .c_str())
        .def("get_marginal_probability",
             &StateVectorBatched<Prec, Space>::get_marginal_probability,
             "measured_values"_a,
             DocString()
                 .desc("Get marginal probabilities for specified measurement outcomes.")
                 .arg("measured_values", "list[int]", "Measurement configuration.")
                 .ret("list[float]", "Probabilities for each state in batch.")
                 .build_as_google_style()
                 .c_str())
        // Entropy and sampling methods
        .def("get_entropy",
             &StateVectorBatched<Prec, Space>::get_entropy,
             DocString()
                 .desc("Calculate von Neumann entropy for each state.")
                 .ret("list[float]", "Entropy values for each state.")
                 .build_as_google_style()
                 .c_str())
        .def(
            "sampling",
            [](const StateVectorBatched<Prec, Space>& states,
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
             &StateVectorBatched<Prec, Space>::add_state_vector_with_coef,
             "coef"_a,
             "states"_a,
             DocString()
                 .desc("Add another batched state vector multiplied by a coefficient.")
                 .arg("coef", "complex", "Coefficient to multiply with states.")
                 .arg("states", "StateVectorBatched", "States to add.")
                 .build_as_google_style()
                 .c_str())
        .def("multiply_coef",
             &StateVectorBatched<Prec, Space>::multiply_coef,
             "coef"_a,
             DocString()
                 .desc("Multiply all states by a coefficient.")
                 .arg("coef", "complex", "Coefficient to multiply.")
                 .build_as_google_style()
                 .c_str())
        // Data access methods
        .def("load",
             &StateVectorBatched<Prec, Space>::load,
             "states"_a,
             DocString()
                 .desc("Load amplitudes for all states in batch.")
                 .arg("states", "list[list[complex]]", "Amplitudes for each state.")
                 .build_as_google_style()
                 .c_str())
        .def("get_amplitudes",
             &StateVectorBatched<Prec, Space>::get_amplitudes,
             DocString()
                 .desc("Get amplitudes of all states in batch.")
                 .ret("list[list[complex]]", "Amplitudes for each state.")
                 .build_as_google_style()
                 .c_str())
        // Copy and string representation
        .def("copy",
             &StateVectorBatched<Prec, Space>::copy,
             DocString()
                 .desc("Create a deep copy of this batched state vector.")
                 .ret("StateVectorBatched", "New copy of the states.")
                 .build_as_google_style()
                 .c_str())
        .def("to_string",
             &StateVectorBatched<Prec, Space>::to_string,
             DocString()
                 .desc("Get string representation of the batched states.")
                 .ret("str", "String representation of states.")
                 .ex(DocString::Code(
                     {">>> states = StateVectorBatched.Haar_random_state(2, 3, False)",
                      ">>> print(states.to_string())",
                      " Qubit Count : 3 ",
                      "Dimension : 8",
                      "--------------------",
                      "Batch_id : 0",
                      "State vector : ",
                      "  000 : (-0.135887,-0.331815)",
                      "  001 : (-0.194471,0.108649)",
                      "  010 : (-0.147649,-0.329848)",
                      "  011 : (-0.131489,0.131093)",
                      "  100 : (-0.262069,0.198882)",
                      "  101 : (-0.0797319,-0.313087)",
                      "  110 : (-0.140573,-0.0577208)",
                      "  111 : (0.181703,0.622905)",
                      "--------------------",
                      "Batch_id : 1",
                      "State vector : ",
                      "  000 : (-0.310841,0.342973)",
                      "  001 : (0.16157,-0.216366)",
                      "  010 : (-0.301031,0.2286)",
                      "  011 : (-0.430187,-0.341108)",
                      "  100 : (0.0126325,0.169034)",
                      "  101 : (0.356303,0.033349)",
                      "  110 : (-0.184462,-0.0361127)",
                      "  111 : (0.224724,-0.160959)"}))
                 .build_as_google_style()
                 .c_str())
        .def("__str__",
             &StateVectorBatched<Prec, Space>::to_string,
             DocString()
                 .desc("Get string representation of the batched states.")
                 .ret("str", "String representation of states.")
                 .build_as_google_style()
                 .c_str())
        // JSON serialization
        .def(
            "to_json",
            [](const StateVectorBatched<Prec, Space>& states) { return Json(states).dump(); },
            DocString()
                .desc("Convert states to JSON string.")
                .ret("str", "JSON representation of states.")
                .ex(DocString::Code(
                    {">>> states = StateVectorBatched.Haar_random_state(2, 3, False)",
                     ">>> print(states.to_json())",
                     "{\"batch_size\":2,\"batched_amplitudes\":[{\"amplitudes\":[{\"imag\":-0."
                     "06388485770655017,\"real\":-0.18444457531249306},{\"imag\":-0."
                     "19976277833680336,\"real\":0.02688995276721736},{\"imag\":-0."
                     "10325202586347756,\"real\":0.34750392103639344},{\"imag\":-0."
                     "08316405642178114,\"real\":-0.13786630724295332},{\"imag\":-0."
                     "12472230847944885,\"real\":0.14554495925352498},{\"imag\":-0."
                     "26280362129148116,\"real\":0.11742521097266628},{\"imag\":-0."
                     "2624948420923217,\"real\":0.020338934511145986},{\"imag\":0."
                     "03692345644121347,\"real\":0.7573990906654825}]},{\"amplitudes\":[{\"imag\":-"
                     "0.042863543360962014,\"real\":0.2002535190582227},{\"imag\":-0."
                     "26105089098208206,\"real\":0.033791318581512894},{\"imag\":-0."
                     "5467139724228703,\"real\":0.23960667554139148},{\"imag\":-0.1008220536735562,"
                     "\"real\":0.3431287916056916},{\"imag\":0.26552531402802715,\"real\":-0."
                     "06501035752577479},{\"imag\":0.11913162732583721,\"real\":0."
                     "47146654843051494},{\"imag\":-0.1877230034941065,\"real\":0."
                     "04062968177663162},{\"imag\":-0.16209817213481867,\"real\":-0."
                     "1737591400014162}]}],\"n_qubits\":3}"}))
                .build_as_google_style()
                .c_str())
        .def(
            "load_json",
            [](StateVectorBatched<Prec, Space>& states, const std::string& str) {
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
