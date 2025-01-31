#pragma once

#include <string_view>
#include <vector>

#include "../constant.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace scaluq {

<<<<<<< HEAD
template <Precision Prec>
class Operator;

template <Precision Prec>
class PauliOperator {
    friend class Operator<Prec>;
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;

public:
    class Data {
        friend class PauliOperator<Prec>;
        friend class Operator<Prec>;
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class Operator;

template <std::floating_point Fp, ExecutionSpace Sp>
class PauliOperator {
    friend class Operator<Fp, Sp>;

public:
    class Data {
        friend class PauliOperator<Fp, Sp>;
        friend class Operator<Fp, Sp>;
>>>>>>> set-space
        std::vector<std::uint64_t> _target_qubit_list, _pauli_id_list;
        ComplexType _coef;
        std::uint64_t _bit_flip_mask, _phase_flip_mask;

    public:
        explicit Data(StdComplex coef = 1.) : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {}

        Data(std::string_view pauli_string, StdComplex coef = 1.);

        Data(const std::vector<std::uint64_t>& target_qubit_list,
             const std::vector<std::uint64_t>& pauli_id_list,
             StdComplex coef = 1.);

        Data(const std::vector<std::uint64_t>& pauli_id_par_qubit, StdComplex coef = 1.);

        Data(std::uint64_t bit_flip_mask, std::uint64_t phase_flip_mask, StdComplex coef = 1.);

        void add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id);

        StdComplex coef() const { return _coef; }
        void set_coef(StdComplex c) { _coef = c; }
        const std::vector<std::uint64_t>& target_qubit_list() const { return _target_qubit_list; }
        const std::vector<std::uint64_t>& pauli_id_list() const { return _pauli_id_list; }
        std::tuple<std::uint64_t, std::uint64_t> get_XZ_mask_representation() const {
            return {_bit_flip_mask, _phase_flip_mask};
        }
    };

private:
    std::shared_ptr<const Data> _ptr;

public:
    enum PauliID : std::uint64_t { I, X, Y, Z };

    explicit PauliOperator(StdComplex coef = 1.) : _ptr(std::make_shared<const Data>(coef)) {}
    explicit PauliOperator(Data data) : _ptr(std::make_shared<const Data>(data)) {}
    PauliOperator(std::string_view pauli_string, StdComplex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_string, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                  const std::vector<std::uint64_t>& pauli_id_list,
                  StdComplex coef = 1.)
        : _ptr(std::make_shared<const Data>(target_qubit_list, pauli_id_list, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit, StdComplex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_id_par_qubit, coef)) {}
    PauliOperator(std::uint64_t bit_flip_mask, std::uint64_t phase_flip_mask, StdComplex coef = 1.)
        : _ptr(std::make_shared<const Data>(bit_flip_mask, phase_flip_mask, coef)) {}

    [[nodiscard]] StdComplex coef() const { return _ptr->coef(); }
    [[nodiscard]] const std::vector<std::uint64_t>& target_qubit_list() const {
        return _ptr->target_qubit_list();
    }
    [[nodiscard]] const std::vector<std::uint64_t>& pauli_id_list() const {
        return _ptr->pauli_id_list();
    }
    [[nodiscard]] std::tuple<std::uint64_t, std::uint64_t> get_XZ_mask_representation() const {
        return _ptr->get_XZ_mask_representation();
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] PauliOperator get_dagger() const;
    [[nodiscard]] std::uint64_t get_qubit_count() const;

<<<<<<< HEAD
    void apply_to_state(StateVector<Prec>& state_vector) const;

    [[nodiscard]] StdComplex get_expectation_value(const StateVector<Prec>& state_vector) const;
    [[nodiscard]] StdComplex get_transition_amplitude(
        const StateVector<Prec>& state_vector_bra, const StateVector<Prec>& state_vector_ket) const;
=======
    void apply_to_state(StateVector<Fp, Sp>& state_vector) const;

    [[nodiscard]] Complex<Fp> get_expectation_value(const StateVector<Fp, Sp>& state_vector) const;
    [[nodiscard]] Complex<Fp> get_transition_amplitude(
        const StateVector<Fp, Sp>& state_vector_bra,
        const StateVector<Fp, Sp>& state_vector_ket) const;
>>>>>>> set-space

    [[nodiscard]] internal::ComplexMatrix get_matrix() const;

    [[nodiscard]] internal::ComplexMatrix get_matrix_ignoring_coef() const;

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const;
    [[nodiscard]] inline PauliOperator operator*(StdComplex target) const {
        return PauliOperator(_ptr->_target_qubit_list, _ptr->_pauli_id_list, _ptr->_coef * target);
    }

    friend void to_json(Json& j, const PauliOperator& pauli) {
        j = Json{{"pauli_string", pauli.get_pauli_string()}, {"coef", pauli.coef()}};
    }
    friend void from_json(const Json& j, PauliOperator& pauli) {
        pauli =
<<<<<<< HEAD
            PauliOperator(j.at("pauli_string").get<std::string>(), j.at("coef").get<StdComplex>());
=======
            PauliOperator(j.at("pauli_string").get<std::string>(), j.at("coef").get<Complex<Fp>>());
>>>>>>> set-space
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_operator_pauli_operator_hpp(nb::module_& m) {
<<<<<<< HEAD
    nb::enum_<typename PauliOperator<Prec>::PauliID>(m, "PauliID")
        .value("I", PauliOperator<Prec>::I)
        .value("X", PauliOperator<Prec>::X)
        .value("Y", PauliOperator<Prec>::Y)
        .value("Z", PauliOperator<Prec>::Z)
        .export_values();

    nb::class_<typename PauliOperator<Prec>::Data>(
=======
    nb::enum_<typename PauliOperator<Fp, Sp>::PauliID>(m, "PauliID")
        .value("I", PauliOperator<Fp, Sp>::I)
        .value("X", PauliOperator<Fp, Sp>::X)
        .value("Y", PauliOperator<Fp, Sp>::Y)
        .value("Z", PauliOperator<Fp, Sp>::Z)
        .export_values();

    nb::class_<typename PauliOperator<Fp, Sp>::Data>(
>>>>>>> set-space
        m, "PauliOperatorData", "Internal data structure for PauliOperator.")
        .def(nb::init<StdComplex>(), "coef"_a = 1., "Initialize data with coefficient.")
        .def(nb::init<std::string_view, StdComplex>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize data with pauli string.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      StdComplex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize data with target qubits and pauli ids.")
        .def(nb::init<const std::vector<std::uint64_t>&, StdComplex>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize data with pauli ids per qubit.")
        .def(nb::init<std::uint64_t, std::uint64_t, StdComplex>(),
             "bit_flip_mask"_a,
             "phase_flip_mask"_a,
             "coef"_a = 1.,
             "Initialize data with bit flip and phase flip masks.")
<<<<<<< HEAD
        .def(nb::init<const typename PauliOperator<Prec>::Data&>(),
             "data"_a,
             "Initialize pauli operator from Data object.")
        .def("add_single_pauli",
             &PauliOperator<Prec>::Data::add_single_pauli,
             "target_qubit"_a,
             "pauli_id"_a,
             "Add a single pauli operation to the data.")
        .def("coef", &PauliOperator<Prec>::Data::coef, "Get the coefficient of the Pauli operator.")
        .def("set_coef",
             &PauliOperator<Prec>::Data::set_coef,
             "c"_a,
             "Set the coefficient of the Pauli operator.")
        .def("target_qubit_list",
             &PauliOperator<Prec>::Data::target_qubit_list,
             "Get the list of target qubits.")
        .def("pauli_id_list",
             &PauliOperator<Prec>::Data::pauli_id_list,
             "Get the list of Pauli IDs.")
        .def("get_XZ_mask_representation",
             &PauliOperator<Prec>::Data::get_XZ_mask_representation,
             "Get the X and Z mask representation as a tuple of vectors.");

    nb::class_<PauliOperator<Prec>>(
=======
        .def(nb::init<const typename PauliOperator<Fp, Sp>::Data&>(),
             "data"_a,
             "Initialize pauli operator from Data object.")
        .def("add_single_pauli",
             &PauliOperator<Fp, Sp>::Data::add_single_pauli,
             "target_qubit"_a,
             "pauli_id"_a,
             "Add a single pauli operation to the data.")
        .def("coef",
             &PauliOperator<Fp, Sp>::Data::coef,
             "Get the coefficient of the Pauli operator.")
        .def("set_coef",
             &PauliOperator<Fp, Sp>::Data::set_coef,
             "c"_a,
             "Set the coefficient of the Pauli operator.")
        .def("target_qubit_list",
             &PauliOperator<Fp, Sp>::Data::target_qubit_list,
             "Get the list of target qubits.")
        .def("pauli_id_list",
             &PauliOperator<Fp, Sp>::Data::pauli_id_list,
             "Get the list of Pauli IDs.")
        .def("get_XZ_mask_representation",
             &PauliOperator<Fp, Sp>::Data::get_XZ_mask_representation,
             "Get the X and Z mask representation as a tuple of vectors.");

    nb::class_<PauliOperator<Fp, Sp>>(
>>>>>>> set-space
        m,
        "PauliOperator",
        "Pauli operator as coef and tensor product of single pauli for each qubit.")
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
             "`pauli_id_list[i]` is applied to `target_qubit_list`-th qubit.")
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
             "`paul_id_per_qubit` is applied to `i`-th qubit.")
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
<<<<<<< HEAD
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
=======
        .def("coef", &PauliOperator<Fp, Sp>::coef, "Get property `coef`.")
        .def("target_qubit_list",
             &PauliOperator<Fp, Sp>::target_qubit_list,
             "Get qubits to be applied pauli.")
        .def("pauli_id_list",
             &PauliOperator<Fp, Sp>::pauli_id_list,
             "Get pauli id to be applied. The order is correspond to the result of "
             "`target_qubit_list`")
        .def("get_XZ_mask_representation",
             &PauliOperator<Fp, Sp>::get_XZ_mask_representation,
>>>>>>> set-space
             "Get single-pauli property as binary integer representation. See description of "
             "`__init__(bit_flip_mask_py: int, phase_flip_mask_py: int, coef: float=1.)` for "
             "details.")
        .def("get_pauli_string",
<<<<<<< HEAD
             &PauliOperator<Prec>::get_pauli_string,
             "Get single-pauli property as string representation. See description of "
             "`__init__(pauli_string: str, coef: float=1.)` for details.")
        .def("get_dagger", &PauliOperator<Prec>::get_dagger, "Get adjoint operator.")
        .def("get_qubit_count",
             &PauliOperator<Prec>::get_qubit_count,
             "Get num of qubits to applied with, when count from 0-th qubit. Subset of $[0, "
             "\\mathrm{qubit_count})$ is the target.")
        .def("apply_to_state", &PauliOperator<Prec>::apply_to_state, "Apply pauli to state vector.")
        .def("get_expectation_value",
             &PauliOperator<Prec>::get_expectation_value,
             "Get expectation value of measuring state vector. $\\bra{\\psi}P\\ket{\\psi}$.")
        .def("get_transition_amplitude",
             &PauliOperator<Prec>::get_transition_amplitude,
=======
             &PauliOperator<Fp, Sp>::get_pauli_string,
             "Get single-pauli property as string representation. See description of "
             "`__init__(pauli_string: str, coef: float=1.)` for details.")
        .def("get_dagger", &PauliOperator<Fp, Sp>::get_dagger, "Get adjoint operator.")
        .def("get_qubit_count",
             &PauliOperator<Fp, Sp>::get_qubit_count,
             "Get num of qubits to applied with, when count from 0-th qubit. Subset of $[0, "
             "\\mathrm{qubit_count})$ is the target.")
        .def("apply_to_state",
             &PauliOperator<Fp, Sp>::apply_to_state,
             "Apply pauli to state vector.")
        .def("get_expectation_value",
             &PauliOperator<Fp, Sp>::get_expectation_value,
             "Get expectation value of measuring state vector. $\\bra{\\psi}P\\ket{\\psi}$.")
        .def("get_transition_amplitude",
             &PauliOperator<Fp, Sp>::get_transition_amplitude,
>>>>>>> set-space
             "Get transition amplitude of measuring state vector. $\\bra{\\chi}P\\ket{\\psi}$.")
        .def(nb::self * nb::self)
        .def(nb::self * StdComplex())
        .def(
            "to_json",
<<<<<<< HEAD
            [](const PauliOperator<Prec>& pauli) { return Json(pauli).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](PauliOperator<Prec>& pauli, const std::string& str) {
=======
            [](const PauliOperator<Fp, Sp>& pauli) { return Json(pauli).dump(); },
            "Information as json style.")
        .def(
            "load_json",
            [](PauliOperator<Fp, Sp>& pauli, const std::string& str) {
>>>>>>> set-space
                pauli = nlohmann::json::parse(str);
            },
            "Read an object from the JSON representation of the Pauli operator.");
}
}  // namespace internal
#endif
}  // namespace scaluq
