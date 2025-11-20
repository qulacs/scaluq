#include <bitset>
#include <scaluq/operator/apply_pauli.hpp>
#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/prec_space.hpp>
#include <scaluq/util/math.hpp>

namespace scaluq {
template <Precision Prec>
PauliOperator<Prec>::PauliOperator(std::string_view pauli_string, StdComplex coef) : _coef(coef) {
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

template <Precision Prec>
PauliOperator<Prec>::PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                                   const std::vector<std::uint64_t>& pauli_id_list,
                                   StdComplex coef)
    : _coef(coef) {
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

template <Precision Prec>
PauliOperator<Prec>::PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit,
                                   StdComplex coef)
    : _coef(coef) {
    for (std::uint64_t i = 0; i < pauli_id_par_qubit.size(); ++i) {
        if (pauli_id_par_qubit[i] != PauliOperator<Prec>::I) {
            add_single_pauli(i, pauli_id_par_qubit[i]);
        }
    }
}

template <Precision Prec>
void PauliOperator<Prec>::add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id) {
    if (target_qubit >= sizeof(std::uint64_t) * 8) {
        throw std::runtime_error("PauliOperator::add_single_pauli: target_qubit is too large");
    }
    if (pauli_id >= 4) {
        throw std::runtime_error("PauliOperator::add_single_pauli: pauli_id is invalid");
    }
    if ((_bit_flip_mask | _phase_flip_mask) >> target_qubit & 1) {
        throw std::runtime_error(
            "PauliOperator::add_single_pauli: You cannot add single pauli twice for "
            "same qubit.");
    }
    if (pauli_id == PauliOperator::X || pauli_id == PauliOperator::Y) {
        _bit_flip_mask |= 1ULL << target_qubit;
    }
    if (pauli_id == PauliOperator::Y || pauli_id == PauliOperator::Z) {
        _phase_flip_mask |= 1ULL << target_qubit;
    }
}

template <Precision Prec>
std::vector<std::uint64_t> PauliOperator<Prec>::target_qubit_list() const {
    return internal::mask_to_vector(_bit_flip_mask | _phase_flip_mask);
}
template <Precision Prec>
std::vector<std::uint64_t> PauliOperator<Prec>::pauli_id_list() const {
    std::vector<std::uint64_t> id_list;
    for (std::uint64_t sub_mask = _bit_flip_mask | _phase_flip_mask; sub_mask;
         sub_mask &= (sub_mask - 1)) {
        std::uint64_t q = std::countr_zero(sub_mask);
        if ((_bit_flip_mask & _phase_flip_mask) >> q & 1) {
            id_list.push_back(PauliID::Y);
        } else if (_bit_flip_mask >> q & 1) {
            id_list.push_back(PauliID::X);
        } else if (_phase_flip_mask >> q & 1) {
            id_list.push_back(PauliID::Z);
        }
    }
    return id_list;
}

template <Precision Prec>
std::string PauliOperator<Prec>::get_pauli_string() const {
    auto target_qubit_list = this->target_qubit_list();
    auto pauli_id_list = this->pauli_id_list();
    std::string res;
    for (std::uint64_t i = 0; i < target_qubit_list.size(); ++i) {
        std::uint64_t target_qubit = target_qubit_list[i];
        std::uint64_t pauli_id = pauli_id_list[i];
        res += [&] {
            switch (pauli_id) {
                case PauliOperator::I:
                    return "I ";
                case PauliOperator::X:
                    return "X ";
                case PauliOperator::Y:
                    return "Y ";
                case PauliOperator::Z:
                    return "Z ";
                default:
                    throw std::runtime_error(
                        "PauliOperator::get_pauli_string: Invalid Pauli ID encountered");
            }
        }();
        res += std::to_string(target_qubit);
        if (i + 1 != target_qubit_list.size()) res += ' ';
    }
    return res;
}

template <Precision Prec>
PauliOperator<Prec> PauliOperator<Prec>::get_dagger() const {
    return PauliOperator(_bit_flip_mask, _phase_flip_mask, scaluq::internal::conj(_coef));
}

template <Precision Prec>
std::vector<Eigen::Triplet<StdComplex>> PauliOperator<Prec>::get_matrix_triplets_ignoring_coef()
    const {
    std::uint64_t bit_mask = 0, phase_mask = 0;
    for (std::uint64_t sub_mask = _bit_flip_mask | _phase_flip_mask, idx = 0; sub_mask;
         sub_mask &= (sub_mask - 1), ++idx) {
        std::uint64_t q = std::countr_zero(sub_mask);
        if (_bit_flip_mask >> q & 1) bit_mask |= (1ULL << idx);
        if (_phase_flip_mask >> q & 1) phase_mask |= (1ULL << idx);
    }
    // count PauliID::Y
    std::uint64_t rot90_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    StdComplex rot =
        std::vector<StdComplex>{1., StdComplex(0, -1), -1., StdComplex(0, 1)}[rot90_count % 4];
    std::vector<Eigen::Triplet<StdComplex>> ret;
    std::uint64_t matrix_dim = 1ULL << std::bit_width(bit_mask | phase_mask);
    ret.reserve(matrix_dim * 2);
    for (std::uint64_t index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1 - 2 * (Kokkos::popcount(index & phase_mask) % 2);
        ret.emplace_back(index, index ^ bit_mask, rot * sign);
    }
    return ret;
}

template <Precision Prec>
std::vector<Eigen::Triplet<StdComplex>> PauliOperator<Prec>::get_full_matrix_triplets_ignoring_coef(
    std::uint64_t n_qubits) const {
    std::uint64_t rot90_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    StdComplex rot =
        std::vector<StdComplex>{1., StdComplex(0, -1), -1., StdComplex(0, 1)}[rot90_count % 4];
    std::vector<Eigen::Triplet<StdComplex>> ret;
    std::uint64_t matrix_dim = 1ULL << n_qubits;
    ret.reserve(matrix_dim * 2);
    for (std::uint64_t index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1 - 2 * (Kokkos::popcount(index & _phase_flip_mask) % 2);
        ret.emplace_back(index, index ^ _bit_flip_mask, rot * sign);
    }
    return ret;
}

template <Precision Prec>
ComplexMatrix PauliOperator<Prec>::get_matrix() const {
    auto triplets = get_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << std::popcount(_bit_flip_mask | _phase_flip_mask);
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) =
            triplets[i].value() * static_cast<StdComplex>(_coef);
    }
    return mat;
}

template <Precision Prec>
ComplexMatrix PauliOperator<Prec>::get_full_matrix(std::uint64_t n_qubits) const {
    auto triplets = get_full_matrix_triplets_ignoring_coef(n_qubits);
    std::uint64_t dim = 1ULL << n_qubits;
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec>
ComplexMatrix PauliOperator<Prec>::get_matrix_ignoring_coef() const {
    auto triplets = get_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << std::popcount(_bit_flip_mask | _phase_flip_mask);
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec>
ComplexMatrix PauliOperator<Prec>::get_full_matrix_ignoring_coef(std::uint64_t n_qubits) const {
    auto triplets = get_full_matrix_triplets_ignoring_coef(n_qubits);
    std::uint64_t dim = 1ULL << n_qubits;
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec>
std::string PauliOperator<Prec>::to_string() const {
    std::stringstream ss;
    ss << coef() << " \"" << get_pauli_string() << "\"";
    return ss.str();
}

template class PauliOperator<internal::Prec>;

}  // namespace scaluq
