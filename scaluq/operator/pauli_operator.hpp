#pragma once

#include <string_view>
#include <vector>

#include "../constant.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "apply_pauli.hpp"

namespace scaluq {

template <std::floating_point Fp>
class Operator;

template <std::floating_point Fp>
class PauliOperator {
    friend class Operator<Fp>;

public:
    class Data {
        friend class PauliOperator<Fp>;
        friend class Operator<Fp>;
        std::vector<std::uint64_t> _target_qubit_list, _pauli_id_list;
        Complex<Fp> _coef;
        std::uint64_t _bit_flip_mask, _phase_flip_mask;

    public:
        explicit Data(Complex<Fp> coef = 1.)
            : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {}
        Data(std::string_view pauli_string, Complex<Fp> coef = 1.)
            : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
            auto ss = std::stringstream(std::string(pauli_string));
            while (1) {
                char pauli;
                std::uint64_t target;
                ss >> pauli;
                if (ss.fail()) break;
                ss >> target;
                if (ss.fail()) {
                    throw std::runtime_error(
                        "PauliOperator::PauliOperator: invalid pauli_string format");
                }
                std::uint64_t pauli_id = [&] {
                    if (pauli == 'I' || pauli == 'i') return PauliOperator::I;
                    if (pauli == 'X' || pauli == 'x') return PauliOperator::X;
                    if (pauli == 'Y' || pauli == 'y') return PauliOperator::Y;
                    if (pauli == 'Z' || pauli == 'z') return PauliOperator::Z;
                    throw std::runtime_error(
                        "PauliOperator::PauliOperator: invalid pauli_string format");
                }();
                if (pauli_id != 0) add_single_pauli(target, pauli_id);
            }
        }

        Data(const std::vector<std::uint64_t>& target_qubit_list,
             const std::vector<std::uint64_t>& pauli_id_list,
             Complex<Fp> coef = 1.)
            : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
            if (target_qubit_list.size() != pauli_id_list.size()) {
                throw std::runtime_error(
                    "PauliOperator::PauliOperator: target_qubit_list must have same size to "
                    "pauli_id_list");
            }
            for (std::uint64_t term_index = 0; term_index < target_qubit_list.size();
                 ++term_index) {
                add_single_pauli(target_qubit_list[term_index], pauli_id_list[term_index]);
            }
        }

        Data(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex<Fp> coef = 1.)
            : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
            for (std::uint64_t i = 0; i < pauli_id_par_qubit.size(); ++i) {
                add_single_pauli(i, pauli_id_par_qubit[i]);
            }
        }

        Data(std::uint64_t bit_flip_mask, std::uint64_t phase_flip_mask, Complex<Fp> coef)
            : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
            for (std::uint64_t target_idx = 0; target_idx < sizeof(std::uint64_t) * 8;
                 target_idx++) {
                bool bit_flip = bit_flip_mask >> target_idx & 1;
                bool phase_flip = phase_flip_mask >> target_idx & 1;
                if (!bit_flip) {
                    if (!phase_flip)
                        continue;
                    else
                        add_single_pauli(target_idx, 3);
                } else {
                    if (!phase_flip)
                        add_single_pauli(target_idx, 1);
                    else {
                        add_single_pauli(target_idx, 2);
                    }
                }
            }
        }

        void add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id) {
            if (target_qubit >= sizeof(std::uint64_t) * 8) {
                throw std::runtime_error(
                    "PauliOperator::Data::add_single_pauli: target_qubit is too large");
            }
            if (pauli_id >= 4) {
                throw std::runtime_error(
                    "PauliOperator::Data::add_single_pauli: pauli_id is invalid");
            }
            _target_qubit_list.push_back(target_qubit);
            _pauli_id_list.push_back(pauli_id);
            if ((_bit_flip_mask | _phase_flip_mask) >> target_qubit & 1) {
                throw std::runtime_error(
                    "PauliOperator::Data::add_single_pauli: You cannot add single pauli twice for "
                    "same "
                    "qubit.");
            }
            if (pauli_id == PauliOperator::X || pauli_id == PauliOperator::Y) {
                _bit_flip_mask |= 1ULL << target_qubit;
            }
            if (pauli_id == PauliOperator::Y || pauli_id == PauliOperator::Z) {
                _phase_flip_mask |= 1ULL << target_qubit;
            }
        }

        Complex<Fp> coef() const { return _coef; }
        void set_coef(Complex<Fp> c) { _coef = c; }
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

    explicit PauliOperator(Complex<Fp> coef = 1.) : _ptr(std::make_shared<const Data>(coef)) {}
    explicit PauliOperator(Data data) : _ptr(std::make_shared<const Data>(data)) {}
    PauliOperator(std::string_view pauli_string, Complex<Fp> coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_string, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                  const std::vector<std::uint64_t>& pauli_id_list,
                  Complex<Fp> coef = 1.)
        : _ptr(std::make_shared<const Data>(target_qubit_list, pauli_id_list, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex<Fp> coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_id_par_qubit, coef)) {}
    PauliOperator(std::uint64_t bit_flip_mask, std::uint64_t phase_flip_mask, Complex<Fp> coef = 1.)
        : _ptr(std::make_shared<const Data>(bit_flip_mask, phase_flip_mask, coef)) {}

    [[nodiscard]] inline Complex<Fp> coef() const { return _ptr->coef(); }
    [[nodiscard]] inline const std::vector<std::uint64_t>& target_qubit_list() const {
        return _ptr->target_qubit_list();
    }
    [[nodiscard]] inline const std::vector<std::uint64_t>& pauli_id_list() const {
        return _ptr->pauli_id_list();
    }
    [[nodiscard]] inline std::tuple<std::uint64_t, std::uint64_t> get_XZ_mask_representation()
        const {
        return _ptr->get_XZ_mask_representation();
    }
    [[nodiscard]] std::string get_pauli_string() const {
        std::stringstream ss;
        std::uint64_t size = _ptr->_target_qubit_list.size();
        if (size == 0) return "";
        for (std::uint64_t term_index = 0; term_index < size; term_index++) {
            if (_ptr->_pauli_id_list[term_index] != 0) {
                ss << "IXYZ"[_ptr->_pauli_id_list[term_index]] << " "
                   << _ptr->_target_qubit_list[term_index] << " ";
            }
        }
        std::string res = ss.str();
        res.pop_back();
        return res;
    }
    [[nodiscard]] inline PauliOperator get_dagger() const {
        return PauliOperator(
            _ptr->_target_qubit_list, _ptr->_pauli_id_list, Kokkos::conj(_ptr->_coef));
    }
    [[nodiscard]] std::uint64_t get_qubit_count() const {
        if (_ptr->_target_qubit_list.empty()) return 0;
        return std::ranges::max(_ptr->_target_qubit_list) + 1;
    }

    void apply_to_state(StateVector<Fp>& state_vector) const {
        if (state_vector.n_qubits() < get_qubit_count()) {
            throw std::runtime_error(
                "PauliOperator::apply_to_state: n_qubits of state_vector is too small to apply the "
                "operator");
        }
        internal::apply_pauli(
            0ULL, _ptr->_bit_flip_mask, _ptr->_phase_flip_mask, _ptr->_coef, state_vector);
    }

    [[nodiscard]] Complex<Fp> get_expectation_value(const StateVector<Fp>& state_vector) const {
        if (state_vector.n_qubits() < get_qubit_count()) {
            throw std::runtime_error(
                "PauliOperator::get_expectation_value: n_qubits of state_vector is too small to "
                "apply "
                "the operator");
        }
        std::uint64_t bit_flip_mask = _ptr->_bit_flip_mask;
        std::uint64_t phase_flip_mask = _ptr->_phase_flip_mask;
        if (bit_flip_mask == 0) {
            Fp res;
            Kokkos::parallel_reduce(
                state_vector.dim(),
                KOKKOS_LAMBDA(std::uint64_t state_idx, Fp & sum) {
                    Fp tmp =
                        (Kokkos::conj(state_vector._raw[state_idx]) * state_vector._raw[state_idx])
                            .real();
                    if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                    sum += tmp;
                },
                res);
            return _ptr->_coef * res;
        }
        std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
        std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
        Complex<Fp> global_phase = internal::PHASE_90ROT<Fp>()[global_phase_90rot_count % 4];
        Fp res;
        Kokkos::parallel_reduce(
            state_vector.dim() >> 1,
            KOKKOS_LAMBDA(std::uint64_t state_idx, Fp & sum) {
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                Fp tmp = Kokkos::real(state_vector._raw[basis_0] *
                                      Kokkos::conj(state_vector._raw[basis_1]) * global_phase * 2.);
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            res);
        return _ptr->_coef * res;
    }
    [[nodiscard]] Complex<Fp> get_transition_amplitude(
        const StateVector<Fp>& state_vector_bra, const StateVector<Fp>& state_vector_ket) const {
        if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
            throw std::runtime_error(
                "state_vector_bra must have same n_qubits to state_vector_ket.");
        }
        if (state_vector_bra.n_qubits() < get_qubit_count()) {
            throw std::runtime_error(
                "PauliOperator::get_expectation_value: n_qubits of state_vector is too small to "
                "apply "
                "the operator");
        }
        std::uint64_t bit_flip_mask = _ptr->_bit_flip_mask;
        std::uint64_t phase_flip_mask = _ptr->_phase_flip_mask;
        if (bit_flip_mask == 0) {
            Complex<Fp> res;
            Kokkos::parallel_reduce(
                state_vector_bra.dim(),
                KOKKOS_LAMBDA(std::uint64_t state_idx, Complex<Fp> & sum) {
                    Complex<Fp> tmp = Kokkos::conj(state_vector_bra._raw[state_idx]) *
                                      state_vector_ket._raw[state_idx];
                    if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                    sum += tmp;
                },
                res);
            Kokkos::fence();
            return _ptr->_coef * res;
        }
        std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
        std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
        Complex<Fp> global_phase = internal::PHASE_90ROT<Fp>()[global_phase_90rot_count % 4];
        Complex<Fp> res;
        Kokkos::parallel_reduce(
            state_vector_bra.dim() >> 1,
            KOKKOS_LAMBDA(std::uint64_t state_idx, Complex<Fp> & sum) {
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                Complex<Fp> tmp1 = Kokkos::conj(state_vector_bra._raw[basis_1]) *
                                   state_vector_ket._raw[basis_0] * global_phase;
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                Complex<Fp> tmp2 = Kokkos::conj(state_vector_bra._raw[basis_0]) *
                                   state_vector_ket._raw[basis_1] * global_phase;
                if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                sum += tmp1 + tmp2;
            },
            res);
        Kokkos::fence();
        return _ptr->_coef * res;
    }

    [[nodiscard]] internal::ComplexMatrix<Fp> get_matrix() const {
        return get_matrix_ignoring_coef() * StdComplex<Fp>(_ptr->_coef);
    }

    [[nodiscard]] internal::ComplexMatrix<Fp> get_matrix_ignoring_coef() const {
        std::uint64_t flip_mask, phase_mask, rot90_count;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, _ptr->_pauli_id_list.size()),
            [&](std::uint64_t i,
                std::uint64_t& f_mask,
                std::uint64_t& p_mask,
                std::uint64_t& rot90_cnt) {
                std::uint64_t pauli_id = _ptr->_pauli_id_list[i];
                if (pauli_id == 1) {
                    f_mask += 1ULL << i;
                } else if (pauli_id == 2) {
                    f_mask += 1ULL << i;
                    p_mask += 1ULL << i;
                    rot90_cnt++;
                } else if (pauli_id == 3) {
                    p_mask += 1ULL << i;
                }
            },
            flip_mask,
            phase_mask,
            rot90_count);
        std::vector<StdComplex<Fp>> rot = {1, Complex<Fp>(0, -1), -1, Complex<Fp>(0, 1)};
        std::uint64_t matrix_dim = 1ULL << _ptr->_pauli_id_list.size();
        internal::ComplexMatrix<Fp> mat = internal::ComplexMatrix<Fp>::Zero(matrix_dim, matrix_dim);
        for (std::uint64_t index = 0; index < matrix_dim; index++) {
            const StdComplex<Fp> sign = 1. - 2. * (Kokkos::popcount(index & phase_mask) % 2);
            mat(index, index ^ flip_mask) = rot[rot90_count % 4] * sign;
        }
        return mat;
    }

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const {
        int extra_90rot_cnt = 0;
        auto x_left = _ptr->_bit_flip_mask & ~_ptr->_phase_flip_mask;
        auto y_left = _ptr->_bit_flip_mask & _ptr->_phase_flip_mask;
        auto z_left = _ptr->_phase_flip_mask & ~_ptr->_bit_flip_mask;
        auto x_right = target._ptr->_bit_flip_mask & ~target._ptr->_phase_flip_mask;
        auto y_right = target._ptr->_bit_flip_mask & target._ptr->_phase_flip_mask;
        auto z_right = target._ptr->_phase_flip_mask & ~target._ptr->_bit_flip_mask;
        extra_90rot_cnt += std::popcount(x_left & y_right);  // XY = iZ
        extra_90rot_cnt += std::popcount(y_left & z_right);  // YZ = iX
        extra_90rot_cnt += std::popcount(z_left & x_right);  // ZX = iY
        extra_90rot_cnt -= std::popcount(x_left & z_right);  // XZ = -iY
        extra_90rot_cnt -= std::popcount(y_left & x_right);  // YX = -iZ
        extra_90rot_cnt -= std::popcount(z_left & y_right);  // ZY = -iX
        extra_90rot_cnt %= 4;
        if (extra_90rot_cnt < 0) extra_90rot_cnt += 4;
        return PauliOperator(
            _ptr->_bit_flip_mask ^ target._ptr->_bit_flip_mask,
            _ptr->_phase_flip_mask ^ target._ptr->_phase_flip_mask,
            _ptr->_coef * target._ptr->_coef * internal::PHASE_90ROT<Fp>()[extra_90rot_cnt]);
    }
    [[nodiscard]] inline PauliOperator operator*(Complex<Fp> target) const {
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
        .def(nb::init<Complex<Fp>>(), "coef"_a = 1., "Initialize data with coefficient.")
        .def(nb::init<std::string_view, Complex<Fp>>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize data with pauli string.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      Complex<Fp>>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize data with target qubits and pauli ids.")
        .def(nb::init<const std::vector<std::uint64_t>&, Complex<Fp>>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize data with pauli ids per qubit.")
        .def(nb::init<std::uint64_t, std::uint64_t, Complex<Fp>>(),
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
        .def(nb::init<Complex<Fp>>(),
             "coef"_a = 1.,
             "Initialize operator which just multiplying coef.")
        .def(nb::init<const std::vector<std::uint64_t>&,
                      const std::vector<std::uint64_t>&,
                      Complex<Fp>>(),
             "target_qubit_list"_a,
             "pauli_id_list"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`pauli_id_list[i]` is applied to `target_qubit_list`-th qubit.")
        .def(nb::init<std::string_view, Complex<Fp>>(),
             "pauli_string"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. If `pauli_string` is `\"X0Y2\"`, Pauli-X is applied to "
             "0-th qubit and Pauli-Y is applied to 2-th qubit. In `pauli_string`, spaces are "
             "ignored.")
        .def(nb::init<const std::vector<std::uint64_t>&, Complex<Fp>>(),
             "pauli_id_par_qubit"_a,
             "coef"_a = 1.,
             "Initialize pauli operator. For each `i`, single pauli correspond to "
             "`paul_id_per_qubit` is applied to `i`-th qubit.")
        .def(nb::init<std::uint64_t, std::uint64_t, Complex<Fp>>(),
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
        .def("get_XZ_mask_representation",
             &PauliOperator::get_XZ_mask_representation,
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
        .def(nb::self * Complex<Fp>());
}
}  // namespace internal
#endif
}  // namespace scaluq
