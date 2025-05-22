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

template <Precision Prec, ExecutionSpace Space>
class StateVector {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    using FloatType = internal::Float<Prec>;
    using ComplexType = internal::Complex<Prec>;
    using ExecutionSpaceType = internal::SpaceType<Space>;

public:
    static constexpr std::uint64_t UNMEASURED = 2;
    Kokkos::View<ComplexType*, ExecutionSpaceType> _raw;
    StateVector() = default;
    StateVector(std::uint64_t n_qubits);
    StateVector(Kokkos::View<ComplexType*, ExecutionSpaceType> view);
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at(std::uint64_t index, StdComplex c);

    /**
     * @attention Very slow. You should use get_amplitudes() instead if you can.
     */
    [[nodiscard]] StdComplex get_amplitude_at(std::uint64_t index);

    [[nodiscard]] static StateVector Haar_random_state(std::uint64_t n_qubits,
                                                       std::uint64_t seed = std::random_device()());
    [[nodiscard]] static StateVector uninitialized_state(std::uint64_t n_qubits);

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);
    void set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::vector<StdComplex> get_amplitudes() const;

    [[nodiscard]] double get_squared_norm() const;

    void normalize();

    [[nodiscard]] double get_zero_probability(std::uint64_t target_qubit_index) const;

    [[nodiscard]] double get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] double get_entropy() const;

    void add_state_vector_with_coef(StdComplex coef, const StateVector& state);
    void multiply_coef(StdComplex coef);

    [[nodiscard]] std::vector<std::uint64_t> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    void load(const std::vector<StdComplex>& other);

    [[nodiscard]] StateVector copy() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state) {
        os << state.to_string();
        return os;
    }

    [[nodiscard]] std::string to_string() const;

    friend void to_json(Json& j, const StateVector& state) {
        j = Json{{"n_qubits", state._n_qubits}, {"amplitudes", state.get_amplitudes()}};
    }
    friend void from_json(const Json& j, StateVector& state) {
        state = StateVector(j.at("n_qubits").get<std::uint64_t>());
        state.load(j.at("amplitudes").get<std::vector<StdComplex>>());
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_state_state_vector_hpp(nb::module_& m) {
    nb::class_<StateVector<Prec, Space>>(
        m,
        "StateVector",
        DocString()
            .desc("Vector representation of quantum state.")
            .desc("Qubit index is "
                  "start from 0. If the i-th value of the vector is "
                  "$a_i$, the state is $\\sum_i a_i \\ket{i}$.")
            .desc("Given `n_qubits: int`, construct with bases "
                  "$\\ket{0\\dots 0}$ holding `n_qubits` number of qubits.")
            .ex(DocString::Code({">>> state1 = StateVector(2)",
                                 ">>> print(state1)",
                                 " *** Quantum State ***",
                                 " * Qubit Count : 2",
                                 " * Dimension   : 4",
                                 " * State vector : ",
                                 "00: (1,0)",
                                 "01: (0,0)",
                                 "10: (0,0)",
                                 "11: (0,0)",
                                 ""}))
            .build_as_google_style()
            .c_str())
        .def(nb::init<std::uint64_t>(),
             "n_qubits"_a,
             DocString()
                 .desc("Construct with specified number of qubits.")
                 .desc("Vector is initialized with computational "
                       "basis $\\ket{0\\dots0}$.")
                 .arg("n_qubits", "int", "number of qubits")
                 .ex(DocString::Code({">>> state1 = StateVector(2)",
                                      ">>> print(state1)",
                                      " *** Quantum State ***",
                                      " * Qubit Count : 2",
                                      " * Dimension   : 4",
                                      " * State vector : ",
                                      "00: (1,0)",
                                      "01: (0,0)",
                                      "10: (0,0)",
                                      "11: (0,0)"}))
                 .build_as_google_style()
                 .c_str())
        .def_static(
            "Haar_random_state",
            [](std::uint64_t n_qubits, std::optional<std::uint64_t> seed) {
                return StateVector<Prec, Space>::Haar_random_state(
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
        .def_static("uninitialized_state",
                    &StateVector<Prec, Space>::uninitialized_state,
                    "n_qubits"_a,
                    DocString()
                        .desc("Construct :class:`StateVector` without initializing.")
                        .arg("n_qubits", "int", "number of qubits")
                        .build_as_google_style()
                        .c_str())
        .def("set_amplitude_at",
             &StateVector<Prec, Space>::set_amplitude_at,
             "index"_a,
             "value"_a,
             DocString()
                 .desc("Manually set amplitude at one index.")
                 .arg("index",
                      "int",
                      "index of state vector",
                      "This is read as binary.k-th bit of index represents k-th qubit.")
                 .arg("value", "complex", "amplitude value to set at index")
                 .ex(DocString::Code({">>> state = StateVector(2)",
                                      ">>> state.get_amplitudes()",
                                      "[(1+0j), 0j, 0j, 0j]",
                                      ">>> state.set_amplitude_at(2, 3+1j)",
                                      ">>> state.get_amplitudes()",
                                      "[(1+0j), 0j, (3+1j), 0j]"}))
                 .note("If you want to get amplitudes at all indices, you should use "
                       ":meth:`.load`.")
                 .build_as_google_style()
                 .c_str())
        .def("get_amplitude_at",
             &StateVector<Prec, Space>::get_amplitude_at,
             "index"_a,
             DocString()
                 .desc("Get amplitude at one index.")
                 .arg("index",
                      "int",
                      "index of state vector",
                      "This is read as binary. k-th bit of index represents k-th qubit.")
                 .ret("complex", "Amplitude at specified index")
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
             &StateVector<Prec, Space>::set_zero_state,
             DocString()
                 .desc("Initialize with computational basis $\\ket{00\\dots0}$.")
                 .ex(DocString::Code{">>> state = StateVector.Haar_random_state(2)",
                                     ">>> state.get_amplitudes()",
                                     "[(-0.05726462181150916+0.3525270165415515j), "
                                     "(0.1133709060491142+0.3074930854078303j), "
                                     "(0.03542174692996924+0.18488950377672345j), "
                                     "(0.8530024105558827+0.04459332470844164j)]",
                                     ">>> state.set_zero_state()",
                                     ">>> state.get_amplitudes()",
                                     "[(1+0j), 0j, 0j, 0j]"})
                 .build_as_google_style()
                 .c_str())
        .def("set_zero_norm_state",
             &StateVector<Prec, Space>::set_zero_norm_state,
             DocString()
                 .desc("Initialize with 0 (null vector).")
                 .ex(DocString::Code{">>> state = StateVector(2)",
                                     ">>> state.get_amplitudes()",
                                     "[(1+0j), 0j, 0j, 0j]",
                                     ">>> state.set_zero_norm_state()",
                                     ">>> state.get_amplitudes()",
                                     "[0j, 0j, 0j, 0j]"})
                 .build_as_google_style()
                 .c_str())
        .def("set_computational_basis",
             &StateVector<Prec, Space>::set_computational_basis,
             "basis"_a,
             DocString()
                 .desc("Initialize with computational basis \\ket{\\mathrm{basis}}.")
                 .arg("basis",
                      "int",
                      "basis as integer format ($0 \\leq \\mathrm{basis} \\leq "
                      "2^{\\mathrm{n\\_qubits}}-1$)")
                 .ex(DocString::Code{">>> state = StateVector(2)",
                                     ">>> state.set_computational_basis(0) # |00>",
                                     ">>> state.get_amplitudes()",
                                     "[(1+0j), 0j, 0j, 0j]",
                                     ">>> state.set_computational_basis(1) # |01>",
                                     ">>> state.get_amplitudes()",
                                     "[0j, (1+0j), 0j, 0j]",
                                     ">>> state.set_computational_basis(2) # |10>",
                                     ">>> state.get_amplitudes()",
                                     "[0j, 0j, (1+0j), 0j]",
                                     ">>> state.set_computational_basis(3) # |11>",
                                     ">>> state.get_amplitudes()",
                                     "[0j, 0j, 0j, (1+0j)]"})
                 .build_as_google_style()
                 .c_str())
        .def("get_amplitudes",
             &StateVector<Prec, Space>::get_amplitudes,
             DocString()
                 .desc("Get all amplitudes as `list[complex]`.")
                 .ret("list[complex]", "amplitudes of list with len $2^{\\mathrm{n\\_qubits}}$")
                 .ex(DocString::Code{">>> state = StateVector(2)",
                                     ">>> state.get_amplitudes()",
                                     "[(1+0j), 0j, 0j, 0j]"})
                 .build_as_google_style()
                 .c_str())
        .def("n_qubits",
             &StateVector<Prec, Space>::n_qubits,
             DocString()
                 .desc("Get num of qubits.")
                 .ret("int", "num of qubits")
                 .ex(DocString::Code{">>> state = StateVector(2)", ">>> state.n_qubits()", "2"})
                 .build_as_google_style()
                 .c_str())
        .def("dim",
             &StateVector<Prec, Space>::dim,
             DocString()
                 .desc("Get dimension of the vector ($=2^\\mathrm{n\\_qubits}$).")
                 .ret("int", "dimension of the vector")
                 .ex(DocString::Code{">>> state = StateVector(2)", ">>> state.dim()", "4"})
                 .build_as_google_style()
                 .c_str())
        .def("get_squared_norm",
             &StateVector<Prec, Space>::get_squared_norm,
             DocString()
                 .desc("Get squared norm of the state. $\\braket{\\psi|\\psi}$.")
                 .ret("float", "squared norm of the state")
                 .ex(DocString::Code{">>> v = [1+2j, 3+4j, 5+6j, 7+8j]",
                                     ">>> state = StateVector(2)",
                                     ">>> state.load(v)",
                                     ">>> state.get_squared_norm()",
                                     "204.0"
                                     ">>> sum([abs(a)**2 for a in v])",
                                     "204.0"})
                 .build_as_google_style()
                 .c_str())
        .def("normalize",
             &StateVector<Prec, Space>::normalize,
             DocString()
                 .desc("Normalize state.")
                 .desc("Let $\\braket{\\psi|\\psi} = 1$ by multiplying constant.")
                 .ex(DocString::Code{">>> v = [1+2j, 3+4j, 5+6j, 7+8j]",
                                     ">>> state = StateVector(2)",
                                     ">>> state.load(v)",
                                     ">>> state.normalize()",
                                     ">>> state.get_amplitudes()",
                                     "[(0.07001400420140048+0.14002800840280097j), "
                                     "(0.21004201260420147+0.28005601680560194j), "
                                     "(0.3500700210070024+0.42008402520840293j), "
                                     "(0.4900980294098034+0.5601120336112039j)]",
                                     ">>> norm = state.get_squared_norm()**.5",
                                     ">>> [a / norm for a in v]"
                                     "[(0.07001400420140048+0.14002800840280097j), "
                                     "(0.21004201260420147+0.28005601680560194j), "
                                     "(0.3500700210070024+0.42008402520840293j), "
                                     "(0.4900980294098034+0.5601120336112039j)]"})
                 .build_as_google_style()
                 .c_str())
        .def(
            "get_zero_probability",
            &StateVector<Prec, Space>::get_zero_probability,
            "index"_a,
            DocString()
                .desc("Get the probability to observe $\\ket{0}$ at specified index.")
                .desc("**State must be normalized.**")
                .arg("index", "int", "qubit index to be observed")
                .ret("float", "probability to observe $\\ket{0}$")
                .ex(DocString::Code{">>> v = [1 / 6**.5, 2j / 6**.5 * 1j, -1 / 6**.5, -2j / 6**.5]",
                                    ">>> state = StateVector(2)",
                                    ">>> state.load(v)",
                                    ">>> state.get_zero_probability(0)",
                                    "0.3333333333333334",
                                    ">>> state.get_zero_probability(1)",
                                    "0.8333333333333336",
                                    ">>> abs(v[0])**2+abs(v[2])**2",
                                    "0.3333333333333334",
                                    ">>> abs(v[0])**2+abs(v[1])**2",
                                    "0.8333333333333336"})
                .build_as_google_style()
                .c_str())
        .def("get_marginal_probability",
             &StateVector<Prec, Space>::get_marginal_probability,
             "measured_values"_a,
             DocString()
                 .desc("Get the marginal probability to observe as given.")
                 .desc("**State must be normalized.**")
                 .arg("measured_values",
                      "list[int]",
                      "list with len n_qubits.",
                      "`0`, `1` or :attr:`.UNMEASURED` is allowed for each elements. `0` or `1` "
                      "shows the qubit is observed and the value is got. :attr:`.UNMEASURED` "
                      "shows the the qubit is not observed.")
                 .ret("float", "probability to observe as given")
                 .ex(DocString::Code{
                     ">>> v = [1/4, 1/2, 0, 1/4, 1/4, 1/2, 1/4, 1/2]",
                     "state = StateVector(3)",
                     ">>> state.load(v)",
                     ">>> state.get_marginal_probability([0, 1, StateVector.UNMEASURED])",
                     "0.0625",
                     ">>> abs(v[2])**2 + abs(v[6])**2",
                     "0.0625"})
                 .build_as_google_style()
                 .c_str())
        .def("get_entropy",
             &StateVector<Prec, Space>::get_entropy,
             DocString()
                 .desc("Get the entropy of the vector.")
                 .desc("**State must be normalized.**")
                 .ret("float", "entropy")
                 .ex(DocString::Code{
                     ">>> v = [1/4, 1/2, 0, 1/4, 1/4, 1/2, 1/4, 1/2]",
                     ">>> state = StateVector(3)",
                     ">>> state.load(v)",
                     ">>> state.get_entropy()",
                     "2.5000000000000497",
                     ">>> sum(-abs(a)**2 * math.log2(abs(a)**2) for a in v if a != 0)",
                     "2.5"})
                 .note("The result of this function differs from qulacs. This is because scaluq "
                       "adopted 2 for the base of log in the definition of entropy $\\sum_i -p_i "
                       "\\log p_i$ "
                       "however qulacs adopted e.")
                 .build_as_google_style()
                 .c_str())
        .def("add_state_vector_with_coef",
             &StateVector<Prec, Space>::add_state_vector_with_coef,
             "coef"_a,
             "state"_a,
             DocString()
                 .desc("Add other state vector with multiplying the coef and make superposition.")
                 .desc("$\\ket{\\mathrm{this}}\\leftarrow\\ket{\\mathrm{this}}+\\mathrm{coef} "
                       "\\ket{\\mathrm{state}}$.")
                 .arg("coef", "complex", "coefficient to multiply to `state`")
                 .arg("state", ":class:`StateVector`", "state to be added")
                 .ex(DocString::Code{">>> state1 = StateVector(1)",
                                     ">>> state1.load([1, 2])",
                                     ">>> state2 = StateVector(1)",
                                     ">>> state2.load([3, 4])",
                                     ">>> state1.add_state_vector_with_coef(2j, state2)",
                                     ">>> state1.get_amplitudes()",
                                     "[(1+6j), (2+8j)]"})
                 .build_as_google_style()
                 .c_str())
        .def("multiply_coef",
             &StateVector<Prec, Space>::multiply_coef,
             "coef"_a,
             DocString()
                 .desc("Multiply coef.")
                 .desc("$\\ket{\\mathrm{this}}\\leftarrow\\mathrm{coef}\\ket{\\mathrm{this}}$.")
                 .arg("coef", "complex", "coefficient to multiply")
                 .ex(DocString::Code{">>> state = StateVector(1)",
                                     ">>> state.load([1, 2])",
                                     ">>> state.multiply_coef(2j)",
                                     ">>> state.get_amplitudes()",
                                     "[2j, 4j]"})
                 .build_as_google_style()
                 .c_str())
        .def(
            "sampling",
            [](const StateVector<Prec, Space>& state,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return state.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Sampling state vector independently and get list of computational basis")
                .arg("sampling_count", "int", "how many times to apply sampling")
                .arg("seed",
                     "int | None",
                     true,
                     "random seed",
                     "If not specified, the value from random device is used.")
                .ret("list[int]",
                     "result of sampling",
                     "list of `sampling_count` length. Each element is in "
                     "$[0,2^{\\mathrm{n\\_qubits}})$")
                .ex(DocString::Code{" >>> state = StateVector(2)",
                                    ">>> state.load([1/2, 0, -3**.5/2, 0])",
                                    ">>> state.sampling(8) ",
                                    "[0, 2, 2, 2, 2, 0, 0, 2]"})
                .build_as_google_style()
                .c_str())
        .def(
            "to_string",
            &StateVector<Prec, Space>::to_string,
            DocString()
                .desc("Information as `str`.")
                .ret("str", "information as str")
                .ex(DocString::Code{
                    ">>> state = StateVector(1)",
                    ">>> state.to_string()",
                    R"(' *** Quantum State ***\n * Qubit Count : 1\n * Dimension   : 2\n * State vector : \n0: (1,0)\n1: (0,0)\n')"})
                .build_as_google_style()
                .c_str())
        .def("load",
             &StateVector<Prec, Space>::load,
             "other"_a,
             DocString()
                 .desc("Load amplitudes of `Sequence`")
                 .arg("other",
                      "collections.abc.Sequence[complex]",
                      "list of complex amplitudes with len $2^{\\mathrm{n_qubits}}$")
                 .build_as_google_style()
                 .c_str())
        .def("__str__",
             &StateVector<Prec, Space>::to_string,
             DocString()
                 .desc("Information as `str`.")
                 .desc("Same as :meth:`.to_string()`")
                 .build_as_google_style()
                 .c_str())
        .def_ro_static(
            "UNMEASURED",
            &StateVector<Prec, Space>::UNMEASURED,
            DocString()
                .desc("Constant used for `StateVector::get_marginal_probability` to express the "
                      "the qubit is not measured.")
                .build_as_google_style()
                .c_str())
        .def(
            "to_json",
            [](const StateVector<Prec, Space>& state) { return Json(state).dump(); },
            DocString()
                .desc("Information as json style.")
                .ret("str", "information as json style")
                .ex(DocString::Code{
                    ">>> state = StateVector(1)",
                    ">>> state.to_json()",
                    R"('{"amplitudes":[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0}],"n_qubits":1}')"})
                .build_as_google_style()
                .c_str())
        .def(
            "load_json",
            [](StateVector<Prec, Space>& state, const std::string& str) {
                state = nlohmann::json::parse(str);
            },
            "json_str"_a,
            DocString()
                .desc("Read an object from the JSON representation of the state vector.")
                .build_as_google_style()
                .c_str());
}
}  // namespace internal
#endif
}  // namespace scaluq
