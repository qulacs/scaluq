#include <Eigen/Eigenvalues>
#include <scaluq/operator/operator.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"

namespace scaluq {
template <>
Operator<internal::Prec, internal::Space>::Operator(
    std::vector<PauliOperator<internal::Prec, internal::Space>> terms)
    : _terms(internal::convert_vector_to_view<PauliOperator<internal::Prec, internal::Space>,
                                              internal::Space>(terms, "terms")) {
    for (auto& term : terms) {
        if (term.coef().imag() != 0) {
            _is_hermitian = false;
            break;
        }
    }
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::copy() const {
    Operator<internal::Prec, internal::Space> copy_operator(_terms.size());
    Kokkos::deep_copy(copy_operator._terms, _terms);
    copy_operator._is_hermitian = _is_hermitian;
    return copy_operator;
}

template <>
std::string Operator<internal::Prec, internal::Space>::to_string() const {
    std::stringstream ss;
    auto vec = get_terms();
    for (auto itr = vec.begin(); itr != vec.end(); ++itr) {
        ss << itr->to_string() << "\n";
    }
    return ss.str();
}

template <>
void Operator<internal::Prec, internal::Space>::load(
    const std::vector<PauliOperator<internal::Prec, internal::Space>>& terms) {
    if (terms.size() != _terms.size()) {
        throw std::runtime_error(
            "Operator::load: size of terms does not match the current operator size.");
    }
    auto host_view = internal::wrapped_host_view(terms);
    Kokkos::deep_copy(_terms, host_view);
    _is_hermitian = true;
    for (auto& term : terms) {
        if (term.coef().imag() != 0) {
            _is_hermitian = false;
            break;
        }
    }
}

template <>
Operator<internal::Prec, internal::Space>
Operator<internal::Prec, internal::Space>::uninitialized_operator(std::uint64_t n_terms) {
    Operator<internal::Prec, internal::Space> tmp;
    tmp._terms = Kokkos::View<PauliOperator<internal::Prec, internal::Space>*, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("terms"), n_terms);
    return tmp;
}

template <>
void Operator<internal::Prec, internal::Space>::optimize() {
    // TODO: use Kokkos::UnorderedMap
    std::map<std::tuple<std::uint64_t, std::uint64_t>, ComplexType> pauli_and_coef;
    auto terms_h = get_terms();
    for (const auto& pauli : terms_h) {
        pauli_and_coef[pauli.get_XZ_mask_representation()] += pauli.coef();
    }
    terms_h.clear();
    for (const auto& [mask, coef] : pauli_and_coef) {
        const auto& [x_mask, z_mask] = mask;
        terms_h.emplace_back(x_mask, z_mask, StdComplex(coef));
    }
    *this = Operator<internal::Prec, internal::Space>(terms_h);
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::get_dagger()
    const {
    auto copy_operator =
        Operator<internal::Prec, internal::Space>::uninitialized_operator(_terms.size());
    Kokkos::parallel_for(
        "get_dagger",
        Kokkos::RangePolicy<ExecutionSpaceType>(0, _terms.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { copy_operator._terms(i) = _terms(i).get_dagger(); });
    return copy_operator;
}

template <>
ComplexMatrix Operator<internal::Prec, internal::Space>::get_full_matrix(
    std::uint64_t n_qubits) const {
    std::uint64_t dim = 1ULL << n_qubits;
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (const auto& term : this->get_terms()) {
        auto basic_triplets = term.get_full_matrix_triplets_ignoring_coef(n_qubits);
        for (const auto& triplet : basic_triplets) {
            mat(triplet.row(), triplet.col()) += triplet.value() * term.coef();
        }
    }
    return mat;
}

template <>
void Operator<internal::Prec, internal::Space>::apply_to_state(
    StateVector<internal::Prec, internal::Space>& state_vector) const {
    auto res =
        StateVector<internal::Prec, internal::Space>::uninitialized_state(state_vector.n_qubits());
    res.set_zero_norm_state();
    // TODO: batch でできそう
    auto terms_h = get_terms();
    for (const auto& term : terms_h) {
        StateVector<internal::Prec, internal::Space> tmp = state_vector.copy();
        term.apply_to_state(tmp);
        res.add_state_vector_with_coef(1., tmp);
    }
    state_vector = res;
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::get_expectation_value(
    const StateVector<internal::Prec, internal::Space>& state_vector) const {
    std::uint64_t nterms = _terms.size();
    std::uint64_t dim = state_vector.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        "get_expectation_value",
        Kokkos::MDRangePolicy<internal::SpaceType<internal::Space>, Kokkos::Rank<2>>(
            {0, 0}, {nterms, dim >> 1}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, ComplexType & res_lcl) {
            auto bit_flip_mask = _terms[term_id]._bit_flip_mask;
            auto phase_flip_mask = _terms[term_id]._phase_flip_mask;
            ComplexType coef = _terms[term_id]._coef;
            if (bit_flip_mask == 0) {
                std::uint64_t state_idx1 = state_idx << 1;
                FloatType tmp1 = (scaluq::internal::conj(state_vector._raw[state_idx1]) *
                                  state_vector._raw[state_idx1])
                                     .real();
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                std::uint64_t state_idx2 = state_idx1 | 1;
                FloatType tmp2 = (scaluq::internal::conj(state_vector._raw[state_idx2]) *
                                  state_vector._raw[state_idx2])
                                     .real();
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                FloatType tmp =
                    scaluq::internal::real(state_vector._raw[basis_0] *
                                           scaluq::internal::conj(state_vector._raw[basis_1]) *
                                           global_phase * FloatType(2));
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                res_lcl += coef * tmp;
            }
        },
        res);
    Kokkos::fence();
    return static_cast<StdComplex>(res);
}

template <>
std::vector<StdComplex> Operator<internal::Prec, internal::Space>::get_expectation_value(
    const StateVectorBatched<internal::Prec, internal::Space>& states) const {
    std::uint64_t nterms = _terms.size();
    std::uint64_t dim = states.dim();
    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<internal::Space>> res(
        "expectation_value_res", states.batch_size());
    Kokkos::parallel_for(
        "get_expectation_value",
        Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>(
            internal::SpaceType<internal::Space>(), states.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>::member_type&
                team) {
            ComplexType sum = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(team, nterms, dim >> 1),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    auto bit_flip_mask = _terms[term_id]._bit_flip_mask;
                    auto phase_flip_mask = _terms[term_id]._phase_flip_mask;
                    ComplexType coef = _terms[term_id]._coef;
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        FloatType tmp1 =
                            (scaluq::internal::conj(states._raw(batch_id, state_idx1)) *
                             states._raw(batch_id, state_idx1))
                                .real();
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        FloatType tmp2 =
                            (scaluq::internal::conj(states._raw(batch_id, state_idx2)) *
                             states._raw(batch_id, state_idx2))
                                .real();
                        if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    } else {
                        std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                        std::uint64_t global_phase_90rot_count =
                            Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                        ComplexType global_phase =
                            internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                        std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(state_idx, pivot);
                        std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                        FloatType tmp = scaluq::internal::real(
                            states._raw(batch_id, basis_0) *
                            scaluq::internal::conj(states._raw(batch_id, basis_1)) * global_phase *
                            FloatType(2));
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                        res_lcl += coef * tmp;
                    }
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                res(batch_id) = Kokkos::complex<double>(sum.real(), sum.imag());
            });
        });
    Kokkos::fence();
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::get_transition_amplitude(
    const StateVector<internal::Prec, internal::Space>& state_vector_bra,
    const StateVector<internal::Prec, internal::Space>& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
            "state_vector_ket must be same");
    }
    std::uint64_t nterms = _terms.size();
    std::uint64_t dim = state_vector_bra.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        "get_transition_amplitude",
        Kokkos::MDRangePolicy<internal::SpaceType<internal::Space>, Kokkos::Rank<2>>(
            {0, 0}, {nterms, dim >> 1}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, ComplexType & res_lcl) {
            auto bit_flip_mask = _terms[term_id]._bit_flip_mask;
            auto phase_flip_mask = _terms[term_id]._phase_flip_mask;
            ComplexType coef = _terms[term_id]._coef;
            if (bit_flip_mask == 0) {
                std::uint64_t state_idx1 = state_idx << 1;
                ComplexType tmp1 = (scaluq::internal::conj(state_vector_bra._raw[state_idx1]) *
                                    state_vector_ket._raw[state_idx1]);
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                std::uint64_t state_idx2 = state_idx1 | 1;
                ComplexType tmp2 = (scaluq::internal::conj(state_vector_bra._raw[state_idx2]) *
                                    state_vector_ket._raw[state_idx2]);
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                ComplexType tmp1 = scaluq::internal::conj(state_vector_bra._raw[basis_1]) *
                                   state_vector_ket._raw[basis_0] * global_phase;
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                ComplexType tmp2 = scaluq::internal::conj(state_vector_bra._raw[basis_0]) *
                                   state_vector_ket._raw[basis_1] * global_phase;
                if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            }
        },
        res);
    Kokkos::fence();
    return static_cast<StdComplex>(res);
}

template <>
std::vector<StdComplex> Operator<internal::Prec, internal::Space>::get_transition_amplitude(
    const StateVectorBatched<internal::Prec, internal::Space>& states_bra,
    const StateVectorBatched<internal::Prec, internal::Space>& states_ket) const {
    if (states_bra.n_qubits() != states_ket.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
            "state_vector_ket must be same");
    }
    if (states_bra.batch_size() != states_ket.batch_size()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: batch_size of state_vector_bra and "
            "state_vector_ket must be same");
    }
    std::uint64_t nterms = _terms.size();
    std::uint64_t dim = states_bra.dim();
    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<internal::Space>> results(
        "transition_amplitude_res", states_bra.batch_size());
    Kokkos::parallel_for(
        "get_transition_amplitude",
        Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>(
            internal::SpaceType<internal::Space>(), states_bra.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>::member_type&
                team) {
            ComplexType res = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(team, nterms, (dim >> 1)),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    auto bit_flip_mask = _terms[term_id]._bit_flip_mask;
                    auto phase_flip_mask = _terms[term_id]._phase_flip_mask;
                    ComplexType coef = _terms[term_id]._coef;
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        ComplexType tmp1 =
                            (scaluq::internal::conj(states_bra._raw(batch_id, state_idx1)) *
                             states_ket._raw(batch_id, state_idx1));
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        ComplexType tmp2 =
                            (scaluq::internal::conj(states_bra._raw(batch_id, state_idx2)) *
                             states_ket._raw(batch_id, state_idx2));
                        if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    } else {
                        std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                        std::uint64_t global_phase_90rot_count =
                            Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                        ComplexType global_phase =
                            internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                        std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(state_idx, pivot);
                        std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                        ComplexType tmp1 =
                            scaluq::internal::conj(states_bra._raw(batch_id, basis_1)) *
                            states_ket._raw(batch_id, basis_0) * global_phase;
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        ComplexType tmp2 =
                            scaluq::internal::conj(states_bra._raw(batch_id, basis_0)) *
                            states_ket._raw(batch_id, basis_1) * global_phase;
                        if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    }
                },
                res);
            Kokkos::single(Kokkos::PerTeam(team), [res, batch_id, results]() {
                results(batch_id) = Kokkos::complex<double>(res.real(), res.imag());
            });
        });
    Kokkos::fence();
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::calculate_default_mu() const {
    FloatType mu;
    std::uint64_t nterms = _terms.size();
    Kokkos::parallel_reduce(
        "calculate_default_mu",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(0, nterms),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & res_lcl) {
            res_lcl += internal::abs(_terms(i)._coef.real());
        },
        mu);
    Kokkos::fence();
    return StdComplex(static_cast<double>(mu));
}

template <>
Operator<internal::Prec, internal::Space>::GroundState
Operator<internal::Prec, internal::Space>::solve_ground_state_by_power_method(
    const StateVector<internal::Prec, internal::Space>& initial_state,
    std::uint64_t iter_count,
    std::optional<StdComplex> mu) const {
    if (_terms.size() == 0) {
        throw std::runtime_error(
            "Operator::solve_ground_state_eigenvalue_by_power_method: At least one PauliOperator "
            "is required.");
    }
    std::uint64_t nqubits = initial_state.n_qubits();
    StdComplex mu_realized = mu.value_or(calculate_default_mu());
    auto state = initial_state.copy();
    auto tmp_state = StateVector<internal::Prec, internal::Space>::uninitialized_state(nqubits);
    for (std::uint64_t i = 0; i < iter_count; i++) {
        // |state> <- (A-mu I)|state>
        tmp_state.load(state);
        apply_to_state(state);
        state.add_state_vector_with_coef(-mu_realized, tmp_state);
        state.normalize();
    }
    return {get_expectation_value(state), state};
}

template <>
Operator<internal::Prec, internal::Space>::GroundState
Operator<internal::Prec, internal::Space>::solve_ground_state_by_arnoldi_method(
    const StateVector<internal::Prec, internal::Space>& initial_state,
    std::uint64_t iter_count,
    std::optional<StdComplex> mu) const {
    if (_terms.size() == 0) {
        throw std::runtime_error(
            "Operator::solve_ground_state_eigenvalue_by_power_method: At least one PauliOperator "
            "is required.");
    }
    std::uint64_t nqubits = initial_state.n_qubits();
    StdComplex mu_realized = mu.value_or(calculate_default_mu());
    std::vector<StateVector<internal::Prec, internal::Space>> krylov_space_basis;
    krylov_space_basis.reserve(iter_count + 1);
    krylov_space_basis.push_back(initial_state.copy());
    ComplexMatrix hassenberg_matrix = ComplexMatrix::Zero(iter_count, iter_count);
    for (std::uint64_t i = 0; i < iter_count; i++) {
        // |state> <- (A-muI)|state>
        auto state = krylov_space_basis.back().copy();
        apply_to_state(state);
        state.add_state_vector_with_coef(-mu_realized, krylov_space_basis.back());
        // make |state> orthogonal to others
        for (std::uint64_t j = 0; j <= i; j++) {
            auto coef = internal::inner_product<internal::Prec, internal::Space>(
                krylov_space_basis[j]._raw, state._raw);
            hassenberg_matrix(j, i) = static_cast<StdComplex>(coef);
            state.add_state_vector_with_coef(-coef, krylov_space_basis[j]);
        }
        // normalize |state>
        double norm = std::sqrt(state.get_squared_norm());
        if (i + 1 < iter_count) {
            hassenberg_matrix(i + 1, i) = norm;
        }
        state.multiply_coef(1. / norm);
        krylov_space_basis.push_back(state);
    }
    Eigen::ComplexEigenSolver<ComplexMatrix> solver(hassenberg_matrix);
    if (solver.info() == Eigen::ComputationInfo::NoConvergence) {
        throw std::runtime_error(
            "Operator::solve_ground_state_eigenvalue_by_arnoldi_method: "
            "Eigenvalue solver did not converge. Please specify smaller iter_count.");
    }
    if (solver.info() != Eigen::ComputationInfo::Success) {
        throw std::runtime_error(
            "Operator::solve_ground_state_eigenvalue_by_arnoldi_method: "
            "Eigenvalue solver failed.");
    }
    auto eigenvalues = solver.eigenvalues();
    auto eigenvectors = solver.eigenvectors();
    auto minimum_eigenvalue_index =
        std::ranges::min_element(
            eigenvalues,
            [](const StdComplex& a, const StdComplex& b) { return a.real() < b.real(); }) -
        eigenvalues.begin();
    auto ground_state = StateVector<internal::Prec, internal::Space>::uninitialized_state(nqubits);
    ground_state.set_zero_norm_state();
    for (std::uint64_t i = 0; i < iter_count; i++) {
        ground_state.add_state_vector_with_coef(eigenvectors(i, minimum_eigenvalue_index),
                                                krylov_space_basis[i]);
    }
    ground_state.normalize();
    return {eigenvalues[minimum_eigenvalue_index] + mu_realized, ground_state};
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator*=(
    StdComplex coef) {
    Kokkos::parallel_for(
        "operator*=",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(0, _terms.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _terms(i)._coef *= coef; });
    _is_hermitian &= (coef.imag() == 0);
    return *this;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator*(
    StdComplex coef) const {
    return this->copy() *= coef;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator*(
    const Operator<internal::Prec, internal::Space>& target) const {
    auto ret = Operator<internal::Prec, internal::Space>::uninitialized_operator(
        _terms.size() * target._terms.size());
    std::uint64_t nnz_count = 0;
    Kokkos::parallel_reduce(
        "operator*",
        Kokkos::MDRangePolicy<internal::SpaceType<internal::Space>, Kokkos::Rank<2>>(
            {0, 0}, {_terms.size(), target._terms.size()}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j, std::uint64_t & nnz_lcl) {
            ret._terms(i * target._terms.size() + j) = _terms(i) * target._terms(j);
            if (static_cast<double>(ret._terms(i * target._terms.size() + j)._coef.imag()) == 0.)
                ++nnz_lcl;
        },
        nnz_count);
    ret._is_hermitian = (nnz_count == 0);
    return ret;
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator*=(
    const PauliOperator<internal::Prec, internal::Space>& target) {
    Kokkos::parallel_for(
        "operator*=",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(0, _terms.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _terms(i) *= target; });
    _is_hermitian &= (target.coef().imag() == 0);
    return *this;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator*(
    const PauliOperator<internal::Prec, internal::Space>& target) const {
    return this->copy() *= target;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator+(
    const Operator<internal::Prec, internal::Space>& target) const {
    auto ret = Operator<internal::Prec, internal::Space>::uninitialized_operator(
        _terms.size() + target._terms.size());
    Kokkos::parallel_for(
        "operator+",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(
            0, _terms.size() + target._terms.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) {
            if (i < _terms.size()) {
                ret._terms(i) = _terms(i);
            } else {
                ret._terms(i) = target._terms(i - _terms.size());
            }
        });
    ret._is_hermitian = _is_hermitian && target._is_hermitian;
    return ret;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator+(
    const PauliOperator<internal::Prec, internal::Space>& target) const {
    auto ret = Operator<internal::Prec, internal::Space>::uninitialized_operator(_terms.size() + 1);
    Kokkos::parallel_for(
        "operator+",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(0, _terms.size() + 1),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) {
            if (i < _terms.size()) {
                ret._terms(i) = _terms(i);
            } else {
                ret._terms(i) = target;
            }
        });
    ret._is_hermitian = _is_hermitian && (target.coef().imag() == 0.);
    return ret;
}

template class Operator<internal::Prec, internal::Space>;

}  // namespace scaluq
