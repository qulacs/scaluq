#include "pauli_operator.hpp"

#include "constant.hpp"

PauliOperator::PauliOperator(Complex coef) : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {}

PauliOperator::PauliOperator(std::string_view pauli_string, Complex coef)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    auto ss = std::stringstream(std::string(pauli_string));
    while (!ss.eof()) {
        char pauli;
        UINT target;
        ss >> pauli >> target;
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
                             Complex coef = 1.)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    if (target_qubit_list.size() != pauli_id_list.size()) {
        throw std::runtime_error(
            "PauliOperator::PauliOperator: target_qubit_list must have same size to pauli_id_list");
    }
    for (UINT term_index = 0; term_index < target_qubit_list.size(); ++term_index) {
        add_single_pauli(target_qubit_list[term_index], pauli_id_list[term_index]);
    }
}

PauliOperator::PauliOperator(UINT bit_flip_mask, UINT phase_flip_mask, Complex coef = 1.)
    : _coef(coef), _bit_flip_mask(0), _phase_flip_mask(0) {
    UINT num_y = 0;
    for (UINT target_idx = 0; target_idx < sizeof(UINT) * 8; ++target_idx) {
        if (!(bit_flip_mask >> target_idx & 1)) {
            if (!(phase_flip_mask >> target_idx & 1))
                continue;
            else
                add_single_pauli(target_idx, 3);
        } else {
            if (!(phase_flip_mask >> target_idx & 1))
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
    if ((_bit_flip_mask | _phase_flip_mask) >> target_qubit & 1) {
        throw std::runtime_error(
            "PauliOperator::add_single_pauli: You cannot add single pauli twice for same qubit.");
    }
    if (pauli_id == 1 || pauli_id == 2) _bit_flip_mask ^= 1ULL << target_qubit;
    if (pauli_id == 2 || pauli_id == 3) _phase_flip_mask ^= 1ULL << target_qubit;
}

Complex PauliOperator::get_expectation_value(const StateVector& state_vector) const {
    const auto& amplitudes = state_vector.amplitudes_raw();
    if (_bit_flip_mask == 0) {
        double res;
        Kokkos::parallel_reduce(
            "expectation_value",
            state_vector.dim(),
            KOKKOS_LAMBDA(const UINT& state_idx, double& sum) {
                double tmp = std::norm(amplitudes[state_idx]);
                if (std::popcount(state_idx & _phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            res);
        return _coef * res;
    }
    UINT pivot = sizeof(UINT) * 8 - std::countl_zero(_bit_flip_mask) - 1;
    UINT lower_mask = (1ULL << pivot) - 1;
    UINT upper_mask = ~lower_mask;
    UINT global_phase_90rot_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    Complex global_phase = PHASE_90ROT[global_phase_90rot_count % 4];
    double res;
    Kokkos::parallel_reduce(
        "expectation_value",
        state_vector.dim() >> 1,
        KOKKOS_LAMBDA(const UINT& state_idx, UINT& sum) {
            UINT basis_0 = (state_idx & upper_mask) << 1 | (state_idx & lower_mask);
            UINT basis_1 = basis_0 ^ _bit_flip_mask;
            double tmp =
                std::real(std::conj(amplitudes[basis_0]) * amplitudes[basis_1] * global_phase * 2.);
            if (std::popcount(basis_0 & _phase_flip_mask) & 1) tmp = -tmp;
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
    const auto& amplitudes_bra = state_vector_bra.amplitudes_raw();
    const auto& amplitudes_ket = state_vector_ket.amplitudes_raw();
    if (_bit_flip_mask == 0) {
        Complex res;
        Kokkos::parallel_reduce(
            "expectation_value",
            state_vector_bra.dim(),
            KOKKOS_LAMBDA(const UINT& state_idx, Complex& sum) {
                Complex tmp = std::conj(amplitudes_bra[state_idx]) * amplitudes_ket[state_idx];
                if (std::popcount(state_idx & _phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            res);
        return _coef * res;
    }
    UINT pivot = sizeof(UINT) * 8 - std::countl_zero(_bit_flip_mask) - 1;
    UINT lower_mask = (1ULL << pivot) - 1;
    UINT upper_mask = ~lower_mask;
    UINT global_phase_90rot_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    Complex global_phase = PHASE_90ROT[global_phase_90rot_count % 4];
    Complex res;
    Kokkos::parallel_reduce(
        "expectation_value",
        state_vector_bra.dim() >> 1,
        KOKKOS_LAMBDA(const UINT& state_idx, Complex& sum) {
            UINT basis_0 = (state_idx & upper_mask) << 1 | (state_idx & lower_mask);
            UINT basis_1 = basis_0 ^ _bit_flip_mask;
            Complex tmp1 =
                std::conj(amplitudes_bra[basis_1]) * amplitudes_ket[basis_0] * global_phase;
            if (std::popcount(basis_0 & _phase_flip_mask) & 1) tmp1 = -tmp1;
            Complex tmp2 =
                std::conj(amplitudes_bra[basis_0]) * amplitudes_ket[basis_1] * global_phase;
            if (std::popcount(basis_1 & _phase_flip_mask) & 1) tmp2 = -tmp2;
            sum += tmp1 + tmp2;
        },
        res);
    return _coef * res;
}

PauliOperator PauliOperator::operator*(const PauliOperator& target) const {
    int extra_90rot_cnt = 0;
    auto x_left = _bit_flip_mask & ~_phase_flip_mask;
    auto y_left = _bit_flip_mask & _phase_flip_mask;
    auto z_left = ~_bit_flip_mask & _phase_flip_mask;
    auto x_right = target._bit_flip_mask & ~target._phase_flip_mask;
    auto y_right = target._bit_flip_mask & target._phase_flip_mask;
    auto z_right = ~target._bit_flip_mask & target._phase_flip_mask;
    extra_90rot_cnt += std::popcount(x_left & y_right);  // XY = iZ
    extra_90rot_cnt += std::popcount(y_left & z_right);  // YZ = iX
    extra_90rot_cnt += std::popcount(z_left & x_right);  // ZX = iY
    extra_90rot_cnt += std::popcount(x_left & z_right);  // XZ = -iY
    extra_90rot_cnt += std::popcount(y_left & x_right);  // YX = -iZ
    extra_90rot_cnt += std::popcount(z_left & y_right);  // ZY = -iX
    extra_90rot_cnt %= 4;
    if (extra_90rot_cnt < 0) extra_90rot_cnt += 4;
    return PauliOperator(_bit_flip_mask ^ target._bit_flip_mask,
                         _phase_flip_mask ^ target._phase_flip_mask,
                         _coef * target._coef * PHASE_90ROT[extra_90rot_cnt]);
}

PauliOperator& PauliOperator::operator*=(const PauliOperator& target) {
    *this = *this * target;
    return *this;
};
