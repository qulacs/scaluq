#pragma once

#include <string_view>
#include <vector>

#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "../util/bit_vector.hpp"

namespace scaluq {
class PauliOperator {
    friend class Operator;

public:
    class Data {
        friend class PauliOperator;
        friend class Operator;
        std::vector<std::uint64_t> _target_qubit_list, _pauli_id_list;
        Complex _coef;
        internal::BitVector _bit_flip_mask, _phase_flip_mask;

    public:
        explicit Data(Complex coef = 1.);
        Data(std::string_view pauli_string, Complex coef = 1.);
        Data(const std::vector<std::uint64_t>& target_qubit_list,
             const std::vector<std::uint64_t>& pauli_id_list,
             Complex coef = 1.);
        Data(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex coef = 1.);
        Data(const std::vector<bool>& bit_flip_mask,
             const std::vector<bool>& phase_flip_mask,
             Complex coef);
        void add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id);
        Complex coef() const { return _coef; }
        void set_coef(Complex c) { _coef = c; }
        const std::vector<std::uint64_t>& target_qubit_list() const { return _target_qubit_list; }
        const std::vector<std::uint64_t>& pauli_id_list() const { return _pauli_id_list; }
        std::tuple<std::vector<bool>, std::vector<bool>> get_XZ_mask_representation() const {
            return {_bit_flip_mask, _phase_flip_mask};
        }
    };

private:
    std::shared_ptr<const Data> _ptr;

public:
    enum PauliID : std::uint64_t { I, X, Y, Z };

    explicit PauliOperator(Complex coef = 1.) : _ptr(std::make_shared<const Data>(coef)) {}
    explicit PauliOperator(Data data) : _ptr(std::make_shared<const Data>(data)) {}
    PauliOperator(std::string_view pauli_string, Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_string, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                  const std::vector<std::uint64_t>& pauli_id_list,
                  Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(target_qubit_list, pauli_id_list, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_id_par_qubit, coef)) {}
    PauliOperator(const std::vector<bool>& bit_flip_mask,
                  const std::vector<bool>& phase_flip_mask,
                  Complex coef)
        : _ptr(std::make_shared<const Data>(bit_flip_mask, phase_flip_mask, coef)) {}

    [[nodiscard]] inline Complex coef() const { return _ptr->coef(); }
    [[nodiscard]] inline const std::vector<std::uint64_t>& target_qubit_list() const {
        return _ptr->target_qubit_list();
    }
    [[nodiscard]] inline const std::vector<std::uint64_t>& pauli_id_list() const {
        return _ptr->pauli_id_list();
    }
    [[nodiscard]] inline std::tuple<std::vector<bool>, std::vector<bool>>
    get_XZ_mask_representation() const {
        return _ptr->get_XZ_mask_representation();
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] inline PauliOperator get_dagger() const {
        return PauliOperator(
            _ptr->_target_qubit_list, _ptr->_pauli_id_list, Kokkos::conj(_ptr->_coef));
    }
    [[nodiscard]] std::uint64_t get_qubit_count() const {
        if (_ptr->_target_qubit_list.empty()) return 0;
        return std::ranges::max(_ptr->_target_qubit_list) + 1;
    }

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;
    [[nodiscard]] internal::ComplexMatrix get_matrix() const;
    [[nodiscard]] internal::ComplexMatrix get_matrix_ignoring_coef() const;

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const;
    [[nodiscard]] inline PauliOperator operator*(Complex target) const {
        return PauliOperator(_ptr->_target_qubit_list, _ptr->_pauli_id_list, _ptr->_coef * target);
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_operator_pauli_operator_hpp(nb::module_& m) {
    nb::enum_<PauliOperator::PauliID>(m, "PauliID")
        .value("I", PauliOperator::I)
        .value("X", PauliOperator::X)
        .value("Y", PauliOperator::Y)
        .value("Z", PauliOperator::Z)
        .export_values();

    nb::class_<PauliOperator::Data>(
        m, "PauliOperatorData", "Internal data structure for PauliOperator.")
        .def(nb::init<Complex>(), "coef"_a = 1., "Initialize data with coefficient.")
        .def(nb::init<std::string_view, Complex>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize data with pauli string.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      Complex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize data with target qubits and pauli ids.")
        .def(nb::init<const std::vector<std::uint64_t>&, Complex>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize data with pauli ids per qubit.")
        .def(nb::init<const std::vector<bool>&, const std::vector<bool>&, Complex>(),
             "bit_flip_mask"_a,
             "phase_flip_mask"_a,
             "coef"_a = 1.,
             "Initialize data with bit flip and phase flip masks.")
        .def(nb::init<const PauliOperator::Data&>(),
             "data"_a,
             "Initialize pauli operator from Data object.")
        .def("add_single_pauli",
             &PauliOperator::Data::add_single_pauli,
             "target_qubit"_a,
             "pauli_id"_a,
             "Add a single pauli operation to the data.")
        .def("coef", &PauliOperator::Data::coef, "Get the coefficient of the Pauli operator.")
        .def("set_coef",
             &PauliOperator::Data::set_coef,
             "c"_a,
             "Set the coefficient of the Pauli operator.")
        .def("target_qubit_list",
             &PauliOperator::Data::target_qubit_list,
             "Get the list of target qubits.")
        .def("pauli_id_list", &PauliOperator::Data::pauli_id_list, "Get the list of Pauli IDs.")
        .def("get_XZ_mask_representation",
             &PauliOperator::Data::get_XZ_mask_representation,
             "Get the X and Z mask representation as a tuple of vectors.");

    nb::class_<PauliOperator>(
        m,
        "PauliOperator",
        "Pauli operator as coef and tensor product of single pauli for each qubit.")
        .def(nb::init<Complex>(), "coef"_a = 1., "Initialize operator which just multiplying coef.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      Complex>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`pauli_id_list[i]` is applied to `target_qubit_list`-th qubit.")
        .def(nb::init<std::string_view, Complex>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. If `pauli_string` is `\"X0Y2\"`, Pauli-X is applied to "
             "0-th qubit and Pauli-Y is applied to 2-th qubit. In `pauli_string`, spaces are "
             "ignored.")
        .def(nb::init<const std::vector<std::uint64_t>&, Complex>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`paul_id_per_qubit` is applied to `i`-th qubit.")
        .def(
            "__init__",
            [](PauliOperator* t,
               nb::int_ bit_flip_mask_py,
               nb::int_ phase_flip_mask_py,
               Complex coef) {
                internal::BitVector bit_flip_mask(0), phase_flip_mask(0);
                const nb::int_ mask(~0ULL);
                auto& bit_flip_raw = bit_flip_mask.data_raw();
                assert(bit_flip_raw.empty());
                while (bit_flip_mask_py > nb::int_(0)) {
                    bit_flip_raw.push_back((std::uint64_t)nb::int_(bit_flip_mask_py & mask));
                    bit_flip_mask_py >>= nb::int_(64);
                }
                auto& phase_flip_raw = phase_flip_mask.data_raw();
                assert(phase_flip_raw.empty());
                while (phase_flip_mask_py > nb::int_(0)) {
                    phase_flip_raw.push_back((std::uint64_t)nb::int_(phase_flip_mask_py & mask));
                    phase_flip_mask_py >>= nb::int_(64);
                }
                new (t) PauliOperator(bit_flip_mask, phase_flip_mask, coef);
            },
            "bit_flip_mask"_a,
            "phase_flip_mask"_a,
            "coef"_a = 1.,
            "Initialize pauli operator. For each `i`, single pauli applied to `i`-th qubit is got "
            "from `i-th` bit of `bit_flip_mask` and `phase_flip_mask` as follows.\n\n.. "
            "csv-table::\n\n    \"bit_flip\",\"phase_flip\",\"pauli\"\n    \"0\",\"0\",\"I\"\n    "
            "\"0\",\"1\",\"Z\"\n    \"1\",\"0\",\"X\"\n    \"1\",\"1\",\"Y\"")
        .def("coef", &PauliOperator::coef, "Get property `coef`.")
        .def("target_qubit_list",
             &PauliOperator::target_qubit_list,
             "Get qubits to be applied pauli.")
        .def("pauli_id_list",
             &PauliOperator::pauli_id_list,
             "Get pauli id to be applied. The order is correspond to the result of "
             "`target_qubit_list`")
        .def(
            "get_XZ_mask_representation",
            [](const PauliOperator& pauli) {
                const auto [x_mask, z_mask] = pauli.get_XZ_mask_representation();
                nb::int_ x_mask_py(0);
                for (std::uint64_t i = 0; i < x_mask.size(); ++i) {
                    x_mask_py |= nb::int_(x_mask[i]) << nb::int_(i);
                }
                nb::int_ z_mask_py(0);
                for (std::uint64_t i = 0; i < z_mask.size(); ++i) {
                    z_mask_py |= nb::int_(z_mask[i]) << nb::int_(i);
                }
                return std::make_tuple(x_mask_py, z_mask_py);
            },
            "Get single-pauli property as binary integer representation. See description of "
            "`__init__(bit_flip_mask_py: int, phase_flip_mask_py: int, coef: float=1.)` for "
            "details.")
        .def("get_pauli_string",
             &PauliOperator::get_pauli_string,
             "Get single-pauli property as string representation. See description of "
             "`__init__(pauli_string: str, coef: float=1.)` for details.")
        .def("get_dagger", &PauliOperator::get_dagger, "Get adjoint operator.")
        .def("get_qubit_count",
             &PauliOperator::get_qubit_count,
             "Get num of qubits to applied with, when count from 0-th qubit. Subset of $[0, "
             "\\mathrm{qubit_count})$ is the target.")
        .def("apply_to_state", &PauliOperator::apply_to_state, "Apply pauli to state vector.")
        .def("get_expectation_value",
             &PauliOperator::get_expectation_value,
             "Get expectation value of measuring state vector. $\\bra{\\psi}P\\ket{\\psi}$.")
        .def("get_transition_amplitude",
             &PauliOperator::get_transition_amplitude,
             "Get transition amplitude of measuring state vector. $\\bra{\\chi}P\\ket{\\psi}$.")
        .def(nb::self * nb::self)
        .def(nb::self * Complex());
}
}  // namespace internal
#endif
}  // namespace scaluq
