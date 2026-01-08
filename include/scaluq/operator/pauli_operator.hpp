#pragma once

#include <string_view>
#include <vector>

#include "../constant.hpp"
#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"
#include "apply_pauli.hpp"

namespace scaluq {

template <Precision Prec>
struct PauliOperator {
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;
    ComplexType _coef;
    std::uint64_t _bit_flip_mask = 0, _phase_flip_mask = 0;

public:
    enum PauliID : std::uint64_t { I, X, Y, Z };

    KOKKOS_FUNCTION explicit PauliOperator(StdComplex coef = 1.)
        : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {}
    PauliOperator(std::string_view pauli_string, StdComplex coef = 1.);
    PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                  const std::vector<std::uint64_t>& pauli_id_list,
                  StdComplex coef = 1.);
    PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit, StdComplex coef = 1.);
    KOKKOS_FUNCTION PauliOperator(std::uint64_t bit_flip_mask,
                                  std::uint64_t phase_flip_mask,
                                  StdComplex coef = 1.)
        : _coef(coef), _bit_flip_mask(bit_flip_mask), _phase_flip_mask(phase_flip_mask) {}

    void set_coef(StdComplex c) { _coef = c; }
    [[nodiscard]] StdComplex coef() const { return _coef; }
    [[nodiscard]] std::vector<std::uint64_t> target_qubit_list() const;
    [[nodiscard]] std::vector<std::uint64_t> pauli_id_list() const;
    [[nodiscard]] std::tuple<std::uint64_t, std::uint64_t> get_XZ_mask_representation() const {
        return {_bit_flip_mask, _phase_flip_mask};
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] PauliOperator get_dagger() const;
    [[nodiscard]] std::uint64_t get_qubit_count() const;

    void add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id);

    template <ExecutionSpace Space>
    void apply_to_state(StateVector<Prec, Space>& state_vector) const {
        internal::apply_pauli<Prec, Space>(
            0ULL, 0LL, _bit_flip_mask, _phase_flip_mask, _coef, state_vector);
    }

    template <ExecutionSpace Space>
    [[nodiscard]] StdComplex get_expectation_value(
        const StateVector<Prec, Space>& state_vector) const;
    template <ExecutionSpace Space>
    [[nodiscard]] std::vector<StdComplex> get_expectation_value(
        const StateVectorBatched<Prec, Space>& states) const;
    template <ExecutionSpace Space>
    [[nodiscard]] StdComplex get_transition_amplitude(
        const StateVector<Prec, Space>& state_vector_bra,
        const StateVector<Prec, Space>& state_vector_ket) const;
    template <ExecutionSpace Space>
    [[nodiscard]] std::vector<StdComplex> get_transition_amplitude(
        const StateVectorBatched<Prec, Space>& states_bra,
        const StateVectorBatched<Prec, Space>& states_ket) const;

    [[nodiscard]] ComplexMatrix get_matrix() const;
    [[nodiscard]] ComplexMatrix get_matrix_ignoring_coef() const;
    [[nodiscard]] ComplexMatrix get_full_matrix(std::uint64_t n_qubits) const;
    [[nodiscard]] ComplexMatrix get_full_matrix_ignoring_coef(std::uint64_t n_qubits) const;
    [[nodiscard]] std::vector<Eigen::Triplet<StdComplex>> get_matrix_triplets_ignoring_coef() const;
    [[nodiscard]] std::vector<Eigen::Triplet<StdComplex>> get_full_matrix_triplets_ignoring_coef(
        std::uint64_t n_qubits) const;

    [[nodiscard]] KOKKOS_INLINE_FUNCTION PauliOperator
    operator*(const PauliOperator& target) const {
        int extra_90rot_cnt = 0;
        auto x_left = _bit_flip_mask & ~_phase_flip_mask;
        auto y_left = _bit_flip_mask & _phase_flip_mask;
        auto z_left = _phase_flip_mask & ~_bit_flip_mask;
        auto x_right = target._bit_flip_mask & ~target._phase_flip_mask;
        auto y_right = target._bit_flip_mask & target._phase_flip_mask;
        auto z_right = target._phase_flip_mask & ~target._bit_flip_mask;
        extra_90rot_cnt += Kokkos::popcount(x_left & y_right);  // XY = iZ
        extra_90rot_cnt += Kokkos::popcount(y_left & z_right);  // YZ = iX
        extra_90rot_cnt += Kokkos::popcount(z_left & x_right);  // ZX = iY
        extra_90rot_cnt -= Kokkos::popcount(x_left & z_right);  // XZ = -iY
        extra_90rot_cnt -= Kokkos::popcount(y_left & x_right);  // YX = -iZ
        extra_90rot_cnt -= Kokkos::popcount(z_left & y_right);  // ZY = -iX
        extra_90rot_cnt %= 4;
        if (extra_90rot_cnt < 0) extra_90rot_cnt += 4;
        return PauliOperator(_bit_flip_mask ^ target._bit_flip_mask,
                             _phase_flip_mask ^ target._phase_flip_mask,
                             _coef * target._coef * internal::PHASE_90ROT<Prec>()[extra_90rot_cnt]);
    }
    [[nodiscard]] KOKKOS_INLINE_FUNCTION PauliOperator operator*(StdComplex coef) const {
        return PauliOperator(_bit_flip_mask, _phase_flip_mask, _coef * coef);
    }
    KOKKOS_INLINE_FUNCTION PauliOperator& operator*=(const PauliOperator& target) {
        *this = *this * target;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION PauliOperator& operator*=(StdComplex coef) {
        *this = *this * coef;
        return *this;
    }

    [[nodiscard]] std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const PauliOperator& pauli) {
        return os << pauli.to_string();
    }

    friend void to_json(Json& j, const PauliOperator& pauli) {
        j = Json{{"pauli_string", pauli.get_pauli_string()}, {"coef", pauli.coef()}};
    }
    friend void from_json(const Json& j, PauliOperator& pauli) {
        pauli =
            PauliOperator(j.at("pauli_string").get<std::string>(), j.at("coef").get<StdComplex>());
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_operator_pauli_operator_hpp(nb::module_& m) {
    nb::class_<PauliOperator<Prec>>(
        m,
        "PauliOperator",
        DocString()
            .desc("Pauli operator as coef and tensor product of single pauli for each qubit.")
            .desc("Given `coef: complex`, Initialize operator which just multiplying coef.")
            .desc("Given `target_qubit_list: list[int], pauli_id_list: "
                  "list[int], coef: complex`, Initialize pauli operator. For "
                  "each `i`, single pauli correspond to `pauli_id_list[i]` is applied to "
                  "`target_qubit_list[i]`-th qubit.")
            .desc("Given `pauli_string: str, coef: complex`, Initialize pauli "
                  "operator. For each `i`, single pauli correspond to `pauli_id_list[i]` is "
                  "applied to `target_qubit_list[i]`-th qubit.")
            .desc("Given `pauli_id_par_qubit: list[int], coef: complex`, "
                  "Initialize pauli operator. For each `i`, single pauli correspond to "
                  "`paul_id_per_qubit[i]` is applied to `i`-th qubit.")
            .desc("Given `bit_flip_mask: int, phase_flip_mask: int, coef: "
                  "complex`, Initialize pauli operator. For each `i`, single pauli applied to "
                  "`i`-th qubit is got from `i-th` bit of `bit_flip_mask` and `phase_flip_mask` as "
                  "follows.\n\n.. "
                  "csv-table::\n\n    \"bit_flip\",\"phase_flip\",\"pauli\"\n    "
                  "\"0\",\"0\",\"I\"\n    "
                  "\"0\",\"1\",\"Z\"\n    \"1\",\"0\",\"X\"\n    \"1\",\"1\",\"Y\"")
            .ex(DocString::Code(
                {">>> pauli = PauliOperator(\"X 3 Y 2\")",
                 ">>> print(pauli.to_json())",
                 "{\"coef\":{\"imag\":0.0,\"real\":1.0},\"pauli_string\":\"Y 2 X 3\"}"}))
            .build_as_google_style()
            .c_str())
        .def(nb::init<StdComplex>(),
             "coef"_a = 1.,
             "Initialize operator which just multiplying coef.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      StdComplex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`pauli_id_list[i]` is applied to `target_qubit_list[i]`-th qubit.")
        .def(nb::init<std::string_view, StdComplex>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. If `pauli_string` is `\"X0Y2\"`, Pauli-X is applied to "
             "0-th qubit and Pauli-Y is applied to 2-th qubit. In `pauli_string`, spaces are "
             "ignored.")
        .def(nb::init<const std::vector<std::uint64_t>&, StdComplex>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`paul_id_per_qubit[i]` is applied to `i`-th qubit.")
        .def(nb::init<std::uint64_t, std::uint64_t, StdComplex>(),
             "bit_flip_mask"_a,
             "phase_flip_mask"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli applied to `i`-th qubit is "
             "got "
             "from `i-th` bit of `bit_flip_mask` and `phase_flip_mask` as follows.\n\n.. "
             "csv-table::\n\n    \"bit_flip\",\"phase_flip\",\"pauli\"\n    "
             "\"0\",\"0\",\"I\"\n    "
             "\"0\",\"1\",\"Z\"\n    \"1\",\"0\",\"X\"\n    \"1\",\"1\",\"Y\"")
        .def("coef", &PauliOperator<Prec>::coef, "Get property `coef`.")
        .def("target_qubit_list",
             &PauliOperator<Prec>::target_qubit_list,
             "Get qubits to be applied pauli.")
        .def("pauli_id_list",
             &PauliOperator<Prec>::pauli_id_list,
             "Get pauli id to be applied. The order is correspond to the result of "
             "`target_qubit_list`")
        .def("get_XZ_mask_representation",
             &PauliOperator<Prec>::get_XZ_mask_representation,
             "Get single-pauli property as binary integer representation. See description of "
             "`__init__(bit_flip_mask_py: int, phase_flip_mask_py: int, coef: float=1.)` for "
             "details.")
        .def("get_pauli_string",
             &PauliOperator<Prec>::get_pauli_string,
             "Get single-pauli property as string representation. See description of "
             "`__init__(pauli_string: str, coef: float=1.)` for details.")
        .def("get_dagger", &PauliOperator<Prec>::get_dagger, "Get adjoint operator.")
        .def("apply_to_state",
             nb::overload_cast<StateVector<Prec, ExecutionSpace::Host>&>(
                 &PauliOperator<Prec>::template apply_to_state<ExecutionSpace::Host>, nb::const_),
             "state"_a,
             "Apply pauli to state vector.")
        .def("get_expectation_value",
             nb::overload_cast<const StateVector<Prec, ExecutionSpace::Host>&>(
                 &PauliOperator<Prec>::template get_expectation_value<ExecutionSpace::Host>,
                 nb::const_),
             "state"_a,
             "Get expectation value of measuring state vector. $\\bra{\\psi}P\\ket{\\psi}$.")
        .def("get_transition_amplitude",
             nb::overload_cast<const StateVector<Prec, ExecutionSpace::Host>&,
                               const StateVector<Prec, ExecutionSpace::Host>&>(
                 &PauliOperator<Prec>::template get_transition_amplitude<ExecutionSpace::Host>,
                 nb::const_),
             "source"_a,
             "target"_a,
             "Get transition amplitude of measuring state vector. $\\bra{\\chi}P\\ket{\\psi}$.")
#ifdef SCALUQ_ENABLE_CUDA
        .def("get_expectation_value",
             nb::overload_cast<const StateVectorBatched<Prec, ExecutionSpace::Default>&>(
                 &PauliOperator<Prec>::template get_expectation_value<ExecutionSpace::Default>,
                 nb::const_),
             "states"_a,
             "Get expectation values of measuring state vectors. $\\bra{\\psi_i}P\\ket{\\psi_i}$.")
        .def("get_transition_amplitude",
             nb::overload_cast<const StateVectorBatched<Prec, ExecutionSpace::Default>&,
                               const StateVectorBatched<Prec, ExecutionSpace::Default>&>(
                 &PauliOperator<Prec>::template get_transition_amplitude<ExecutionSpace::Default>,
                 nb::const_),
             "states_source"_a,
             "states_target"_a,
             "Get transition amplitudes of measuring state vectors. "
             "$\\bra{\\chi_i}P\\ket{\\psi_i}$.")
        .def(
            "apply_to_state",
            nb::overload_cast<StateVector<Prec, ExecutionSpace::Default>&>(
                &PauliOperator<Prec>::template apply_to_state<ExecutionSpace::Default>, nb::const_),
            "state"_a,
            "Apply pauli to state vector.")
#endif
        .def("get_matrix",
             &PauliOperator<Prec>::get_matrix,
             "Get matrix representation of the PauliOperator. Tensor product is applied from "
             "$(n-1)$ -th qubit to $0$ -th qubit. Only the X, Y, and Z components "
             "are taken into account in the result.")
        .def("get_full_matrix",
             &PauliOperator<Prec>::get_full_matrix,
             "n_qubits"_a,
             "Get matrix representation of the PauliOperator. Tensor product is applied from "
             "$(n-1)$ -th qubit to $0$ -th qubit.")
        .def("get_matrix_ignoring_coef",
             &PauliOperator<Prec>::get_matrix_ignoring_coef,
             "Get matrix representation of the PauliOperator, but with forcing `coef=1.`Only the "
             "X, Y, and Z components are taken into account in the result.")
        .def("get_full_matrix_ignoring_coef",
             &PauliOperator<Prec>::get_full_matrix_ignoring_coef,
             "n_qubits"_a,
             "Get matrix representation of the PauliOperator, but with forcing `coef=1.`")
        .def(nb::self * nb::self)
        .def(nb::self * StdComplex())
        .def(
            "to_json",
            [](const PauliOperator<Prec>& pauli) { return Json(pauli).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](PauliOperator<Prec>& pauli, const std::string& str) {
                pauli = nlohmann::json::parse(str);
            },
            "json_str"_a,
            "Read an object from the JSON representation of the Pauli operator.")
        .def("to_string",
             &PauliOperator<Prec>::to_string,
             "Get string representation of the Pauli operator.")
        .def("__str__",
             &PauliOperator<Prec>::to_string,
             "Get string representation of the Pauli operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
