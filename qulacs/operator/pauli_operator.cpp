#include "pauli_operator.hpp"

#include "constant.hpp"

namespace qulacs {
PauliOperator::PauliOperator(Complex coef) : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {}

PauliOperator::PauliOperator(std::string_view pauli_string, Complex coef) : _coef(coef) {
    auto ss = std::stringstream(std::string(pauli_string));
    while (1) {
        char pauli;
        UINT target;
        ss >> pauli;
        if (ss.fail()) break;
        ss >> target;
        if (ss.fail()) {
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }
        UINT pauli_id = [&] {
            if (pauli == 'I' || pauli == 'i') return 0;
            if (pauli == 'X' || pauli == 'x') return 1;
            if (pauli == 'Y' || pauli == 'y') return 2;
            if (pauli == 'Z' || pauli == 'z') return 3;
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }();
        if (pauli_id != 0) add_single_pauli(target, pauli_id);
    }
}

PauliOperator::PauliOperator(const std::vector<UINT>& target_qubit_list,
                             const std::vector<UINT>& pauli_id_list,
                             Complex coef)
    : _coef(coef) {
    if (target_qubit_list.size() != pauli_id_list.size()) {
        throw std::runtime_error(
            "PauliOperator::PauliOperator: target_qubit_list must have same size to pauli_id_list");
    }
    for (UINT term_index = 0; term_index < target_qubit_list.size(); ++term_index) {
        add_single_pauli(target_qubit_list[term_index], pauli_id_list[term_index]);
    }
}

PauliOperator::PauliOperator(const BitVector& bit_flip_mask,
                             const BitVector& phase_flip_mask,
                             Complex coef)
    : _coef(coef) {
    UINT num_y = 0;
    UINT max_target = 0;
    if (auto msb = bit_flip_mask.msb(); msb != std::numeric_limits<UINT>::max() && max_target < msb)
        max_target = msb;
    if (auto msb = phase_flip_mask.msb();
        msb != std::numeric_limits<UINT>::max() && max_target < msb)
        max_target = msb;
    for (UINT target_idx = 0; target_idx <= max_target; target_idx++) {
        if (!bit_flip_mask[target_idx]) {
            if (!phase_flip_mask[target_idx])
                continue;
            else
                add_single_pauli(target_idx, 3);
        } else {
            if (!phase_flip_mask[target_idx])
                add_single_pauli(target_idx, 1);
            else {
                add_single_pauli(target_idx, 2);
                ++num_y;
            }
        }
    }
}

std::string PauliOperator::get_pauli_string() const {
    std::stringstream ss;
    UINT size = _target_qubit_list.size();
    if (size == 0) return "";
    for (UINT term_index = 0; term_index < size; term_index++) {
        if (_pauli_id_list[term_index] != 0) {
            ss << "IXYZ"[_pauli_id_list[term_index]] << " " << _target_qubit_list[term_index]
               << " ";
        }
    }
    std::string res = ss.str();
    res.pop_back();
    return res;
}

void PauliOperator::add_single_pauli(UINT target_qubit, UINT pauli_id) {
    _target_qubit_list.push_back(target_qubit);
    _pauli_id_list.push_back(pauli_id);
    if ((_bit_flip_mask | _phase_flip_mask)[target_qubit]) {
        throw std::runtime_error(
            "PauliOperator::add_single_pauli: You cannot add single pauli twice for same qubit.");
    }
    if (pauli_id == 1 || pauli_id == 2) {
        _bit_flip_mask[target_qubit] = true;
    }
    if (pauli_id == 2 || pauli_id == 3) {
        _phase_flip_mask[target_qubit] = true;
    }
}

void PauliOperator::apply_to_state(StateVector& state_vector) const {
    if (state_vector.n_qubits() < get_qubit_count()) {
        throw std::runtime_error(
            "PauliOperator::apply_to_state: n_qubits of state_vector is too small to apply the "
            "operator");
    }
    const auto& amplitudes = state_vector.amplitudes_raw();
    UINT bit_flip_mask = _bit_flip_mask.data_raw()[0];
    UINT phase_flip_mask = _phase_flip_mask.data_raw()[0];
    Complex coef = get_coef();
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            state_vector.dim(), KOKKOS_LAMBDA(const UINT& state_idx) {
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    amplitudes[state_idx] *= -coef;
                } else {
                    amplitudes[state_idx] *= coef;
                }
            });
        return;
    }
    UINT pivot = sizeof(UINT) * 8 - std::countl_zero(bit_flip_mask) - 1;
    UINT lower_mask = (1ULL << pivot) - 1;
    UINT upper_mask = ~lower_mask;
    UINT global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex global_phase = PHASE_90ROT().val[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        state_vector.dim() >> 1, KOKKOS_LAMBDA(const UINT& state_idx) {
            UINT basis_0 = (state_idx & upper_mask) << 1 | (state_idx & lower_mask);
            UINT basis_1 = basis_0 ^ bit_flip_mask;
            Complex tmp1 = amplitudes[basis_0] * global_phase;
            Complex tmp2 = amplitudes[basis_1] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
            amplitudes[basis_0] = tmp1 * coef;
            amplitudes[basis_1] = tmp2 * coef;
        });
}

Complex PauliOperator::get_expectation_value(const StateVector& state_vector) const {
    if (state_vector.n_qubits() < get_qubit_count()) {
        throw std::runtime_error(
            "PauliOperator::get_expectation_value: n_qubits of state_vector is too small to apply "
            "the operator");
    }
    const auto& amplitudes = state_vector.amplitudes_raw();
    UINT bit_flip_mask = _bit_flip_mask.data_raw()[0];
    UINT phase_flip_mask = _phase_flip_mask.data_raw()[0];
    if (bit_flip_mask == 0) {
        double res;
        Kokkos::parallel_reduce(
            state_vector.dim(),
            KOKKOS_LAMBDA(const UINT& state_idx, double& sum) {
                double tmp = (Kokkos::conj(amplitudes[state_idx]) * amplitudes[state_idx]).real();
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            res);
        return _coef * res;
    }
    UINT pivot = sizeof(UINT) * 8 - std::countl_zero(bit_flip_mask) - 1;
    UINT lower_mask = (1ULL << pivot) - 1;
    UINT upper_mask = ~lower_mask;
    UINT global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex global_phase = PHASE_90ROT().val[global_phase_90rot_count % 4];
    double res;
    Kokkos::parallel_reduce(
        state_vector.dim() >> 1,
        KOKKOS_LAMBDA(const UINT& state_idx, double& sum) {
            UINT basis_0 = (state_idx & upper_mask) << 1 | (state_idx & lower_mask);
            UINT basis_1 = basis_0 ^ bit_flip_mask;
            double tmp = Kokkos::real(amplitudes[basis_0] * Kokkos::conj(amplitudes[basis_1]) *
                                      global_phase * 2.);
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
            sum += tmp;
        },
        res);
    return _coef * res;
}

Complex PauliOperator::get_transition_amplitude(const StateVector& state_vector_bra,
                                                const StateVector& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error("state_vector_bra must have same n_qubits to state_vector_ket.");
    }
    if (state_vector_bra.n_qubits() < get_qubit_count()) {
        throw std::runtime_error(
            "PauliOperator::get_expectation_value: n_qubits of state_vector is too small to apply "
            "the operator");
    }
    const auto& amplitudes_bra = state_vector_bra.amplitudes_raw();
    const auto& amplitudes_ket = state_vector_ket.amplitudes_raw();
    UINT bit_flip_mask = _bit_flip_mask.data_raw()[0];
    UINT phase_flip_mask = _phase_flip_mask.data_raw()[0];
    if (bit_flip_mask == 0) {
        Complex res;
        Kokkos::parallel_reduce(
            state_vector_bra.dim(),
            KOKKOS_LAMBDA(const UINT& state_idx, Complex& sum) {
                Complex tmp = Kokkos::conj(amplitudes_bra[state_idx]) * amplitudes_ket[state_idx];
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            res);
        return _coef * res;
    }
    UINT pivot = sizeof(UINT) * 8 - std::countl_zero(bit_flip_mask) - 1;
    UINT lower_mask = (1ULL << pivot) - 1;
    UINT upper_mask = ~lower_mask;
    UINT global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex global_phase = PHASE_90ROT().val[global_phase_90rot_count % 4];
    Complex res;
    Kokkos::parallel_reduce(
        state_vector_bra.dim() >> 1,
        KOKKOS_LAMBDA(const UINT& state_idx, Complex& sum) {
            UINT basis_0 = (state_idx & upper_mask) << 1 | (state_idx & lower_mask);
            UINT basis_1 = basis_0 ^ bit_flip_mask;
            Complex tmp1 =
                Kokkos::conj(amplitudes_bra[basis_1]) * amplitudes_ket[basis_0] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
            Complex tmp2 =
                Kokkos::conj(amplitudes_bra[basis_0]) * amplitudes_ket[basis_1] * global_phase;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
            sum += tmp1 + tmp2;
        },
        res);
    return _coef * res;
}

PauliOperator PauliOperator::operator*(const PauliOperator& target) const {
    int extra_90rot_cnt = 0;
    auto x_left = _bit_flip_mask - _phase_flip_mask;
    auto y_left = _bit_flip_mask & _phase_flip_mask;
    auto z_left = _phase_flip_mask - _bit_flip_mask;
    auto x_right = target._bit_flip_mask - target._phase_flip_mask;
    auto y_right = target._bit_flip_mask & target._phase_flip_mask;
    auto z_right = target._phase_flip_mask - target._bit_flip_mask;
    extra_90rot_cnt += (x_left & y_right).popcount();  // XY = iZ
    extra_90rot_cnt += (y_left & z_right).popcount();  // YZ = iX
    extra_90rot_cnt += (z_left & x_right).popcount();  // ZX = iY
    extra_90rot_cnt -= (x_left & z_right).popcount();  // XZ = -iY
    extra_90rot_cnt -= (y_left & x_right).popcount();  // YX = -iZ
    extra_90rot_cnt -= (z_left & y_right).popcount();  // ZY = -iX
    extra_90rot_cnt %= 4;
    if (extra_90rot_cnt < 0) extra_90rot_cnt += 4;
    return PauliOperator(_bit_flip_mask ^ target._bit_flip_mask,
                         _phase_flip_mask ^ target._phase_flip_mask,
                         _coef * target._coef * PHASE_90ROT().val[extra_90rot_cnt]);
}

PauliOperator& PauliOperator::operator*=(const PauliOperator& target) {
    *this = *this * target;
    return *this;
};

}  // namespace qulacs
