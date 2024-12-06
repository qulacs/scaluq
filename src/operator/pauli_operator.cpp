#include <scaluq/operator/pauli_operator.hpp>

#include "../util/template.hpp"
#include "apply_pauli.hpp"

namespace scaluq {
FLOAT(Fp)
PauliOperator<Fp>::Data::Data(std::string_view pauli_string, Complex<Fp> coef)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    auto ss = std::stringstream(std::string(pauli_string));
    while (1) {
        char pauli;
        std::uint64_t target;
        ss >> pauli;
        if (ss.fail()) break;
        ss >> target;
        if (ss.fail()) {
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }
        std::uint64_t pauli_id = [&] {
            if (pauli == 'I' || pauli == 'i') return PauliOperator::I;
            if (pauli == 'X' || pauli == 'x') return PauliOperator::X;
            if (pauli == 'Y' || pauli == 'y') return PauliOperator::Y;
            if (pauli == 'Z' || pauli == 'z') return PauliOperator::Z;
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }();
        if (pauli_id != 0) add_single_pauli(target, pauli_id);
    }
}

FLOAT(Fp)
PauliOperator<Fp>::Data::Data(const std::vector<std::uint64_t>& target_qubit_list,
                              const std::vector<std::uint64_t>& pauli_id_list,
                              Complex<Fp> coef)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    if (target_qubit_list.size() != pauli_id_list.size()) {
        throw std::runtime_error(
            "PauliOperator::PauliOperator: target_qubit_list must have same size to "
            "pauli_id_list");
    }
    for (std::uint64_t term_index = 0; term_index < target_qubit_list.size(); ++term_index) {
        if (pauli_id_list[term_index] != 0) {
            add_single_pauli(target_qubit_list[term_index], pauli_id_list[term_index]);
        }
    }
}

FLOAT(Fp)
PauliOperator<Fp>::Data::Data(const std::vector<std::uint64_t>& pauli_id_par_qubit,
                              Complex<Fp> coef)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    for (std::uint64_t i = 0; i < pauli_id_par_qubit.size(); ++i) {
        if (pauli_id_par_qubit[i] != PauliOperator<Fp>::I) {
            add_single_pauli(i, pauli_id_par_qubit[i]);
        }
    }
}

FLOAT(Fp)
PauliOperator<Fp>::Data::Data(std::uint64_t bit_flip_mask,
                              std::uint64_t phase_flip_mask,
                              Complex<Fp> coef)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    for (std::uint64_t target_idx = 0; target_idx < sizeof(std::uint64_t) * 8; target_idx++) {
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

FLOAT(Fp)
void PauliOperator<Fp>::Data::add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id) {
    if (target_qubit >= sizeof(std::uint64_t) * 8) {
        throw std::runtime_error(
            "PauliOperator::Data::add_single_pauli: target_qubit is too large");
    }
    if (pauli_id >= 4) {
        throw std::runtime_error("PauliOperator::Data::add_single_pauli: pauli_id is invalid");
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

FLOAT(Fp)
std::string PauliOperator<Fp>::get_pauli_string() const {
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

FLOAT(Fp)
PauliOperator<Fp> PauliOperator<Fp>::get_dagger() const {
    return PauliOperator(_ptr->_target_qubit_list, _ptr->_pauli_id_list, Kokkos::conj(_ptr->_coef));
}

FLOAT(Fp)
std::uint64_t PauliOperator<Fp>::get_qubit_count() const {
    if (_ptr->_target_qubit_list.empty()) return 0;
    return std::ranges::max(_ptr->_target_qubit_list) + 1;
}

FLOAT(Fp)
void PauliOperator<Fp>::apply_to_state(StateVector<Fp>& state_vector) const {
    if (state_vector.n_qubits() < get_qubit_count()) {
        throw std::runtime_error(
            "PauliOperator::apply_to_state: n_qubits of state_vector is too small to apply the "
            "operator");
    }
    internal::apply_pauli(
        0ULL, _ptr->_bit_flip_mask, _ptr->_phase_flip_mask, _ptr->_coef, state_vector);
}

FLOAT(Fp)
Complex<Fp> PauliOperator<Fp>::get_expectation_value(const StateVector<Fp>& state_vector) const {
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
                Fp tmp = (Kokkos::conj(state_vector._raw[state_idx]) * state_vector._raw[state_idx])
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
                                  Kokkos::conj(state_vector._raw[basis_1]) * global_phase * Fp{2});
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
            sum += tmp;
        },
        res);
    return _ptr->_coef * res;
}

FLOAT(Fp)
Complex<Fp> PauliOperator<Fp>::get_transition_amplitude(
    const StateVector<Fp>& state_vector_bra, const StateVector<Fp>& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error("state_vector_bra must have same n_qubits to state_vector_ket.");
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

FLOAT(Fp)
internal::ComplexMatrix<Fp> PauliOperator<Fp>::get_matrix() const {
    return get_matrix_ignoring_coef() * StdComplex<Fp>(_ptr->_coef);
}

FLOAT(Fp)
internal::ComplexMatrix<Fp> PauliOperator<Fp>::get_matrix_ignoring_coef() const {
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

FLOAT(Fp)
PauliOperator<Fp> PauliOperator<Fp>::operator*(const PauliOperator& target) const {
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

FLOAT_DECLARE_CLASS(PauliOperator)

}  // namespace scaluq
