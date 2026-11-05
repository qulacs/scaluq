#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <scaluq/operator/operator.hpp>
#include <scaluq/prec_space.hpp>
#include <scaluq/util/math.hpp>

namespace scaluq {
template <>
Operator<internal::Prec, internal::Space>::Operator(
    std::vector<PauliOperator<internal::Prec>> terms)
    : _terms(internal::convert_vector_to_view<PauliOperator<internal::Prec>, internal::Space>(
          terms, "terms")) {
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
    copy_operator._is_view = false;
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
    const std::vector<PauliOperator<internal::Prec>>& terms) {
    if (terms.size() != _terms.size()) {
        throw std::runtime_error(
            "Operator::load: size of terms does not match the current operator size.");
    }
    auto host_view = internal::wrapped_host_view(terms);
    Kokkos::deep_copy(_terms, host_view);
    _is_hermitian = true;
    for (const auto& term : terms) {
        if (term.coef().imag() != 0) {
            _is_hermitian = false;
            break;
        }
    }
}

template <>
void Operator<internal::Prec, internal::Space>::optimize() {
    if (_is_view) [[unlikely]] {
        throw std::runtime_error(
            "Operator::optimize: this operation is not allowed for a view operator.");
    }
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
    copy_operator._is_view = false;
    copy_operator._is_hermitian = _is_hermitian;
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
    auto states = StateVectorBatched<internal::Prec, internal::Space>::uninitialized_state(
        _terms.size(), state_vector.n_qubits());
    states.set_state_vector(state_vector);
    internal::apply_pauli<internal::Prec, internal::Space>(0, 0, _terms, states);
    state_vector = states.get_reduced_state();
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
                                           global_phase * FloatType(2.0f));
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                res_lcl += coef * tmp;
            }
        },
        res);
    return static_cast<StdComplex>(res);
}

// 各termの期待値からなるベクトルを返す
template <>
std::vector<StdComplex> Operator<internal::Prec, internal::Space>::get_expectation_values(
    const StateVector<internal::Prec, internal::Space>& state_vector) const {
    const std::uint64_t nterms = _terms.size();
    const std::uint64_t dim = state_vector.dim();

    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<internal::Space>> res("expvals",
                                                                                     nterms);

    using ExecSpace = internal::SpaceType<internal::Space>;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename TeamPolicy::member_type;

    Kokkos::parallel_for(
        "get_expectation_values",
        TeamPolicy(static_cast<int>(nterms), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Member& team) {
            const std::uint64_t term_id = static_cast<std::uint64_t>(team.league_rank());

            const auto bit_flip_mask = _terms[term_id]._bit_flip_mask;
            const auto phase_flip_mask = _terms[term_id]._phase_flip_mask;
            const ComplexType coef = _terms[term_id]._coef;

            ComplexType sum = ComplexType(0.0, 0.0);

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, static_cast<int>(dim >> 1)),
                [&](const int state_idx_i, ComplexType& lsum) {
                    const std::uint64_t state_idx = static_cast<std::uint64_t>(state_idx_i);

                    if (bit_flip_mask == 0) {
                        const std::uint64_t state_idx1 = state_idx << 1;
                        FloatType tmp1 = (scaluq::internal::conj(state_vector._raw[state_idx1]) *
                                          state_vector._raw[state_idx1])
                                             .real();
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;

                        const std::uint64_t state_idx2 = state_idx1 | 1;
                        FloatType tmp2 = (scaluq::internal::conj(state_vector._raw[state_idx2]) *
                                          state_vector._raw[state_idx2])
                                             .real();
                        if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;

                        lsum += coef * (tmp1 + tmp2);
                    } else {
                        const std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                        const std::uint64_t global_phase_90rot_count =
                            Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                        const ComplexType global_phase =
                            internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];

                        const std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(state_idx, pivot);
                        const std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                        FloatType tmp = scaluq::internal::real(
                            state_vector._raw[basis_0] *
                            scaluq::internal::conj(state_vector._raw[basis_1]) * global_phase *
                            FloatType(2.0f));

                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;

                        lsum += coef * tmp;
                    }
                },
                sum);

            Kokkos::single(Kokkos::PerTeam(team),
                           [&] { res(term_id) = Kokkos::complex<double>(sum.real(), sum.imag()); });
        });

    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
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
                            FloatType(2.0f));
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                        res_lcl += coef * tmp;
                    }
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                res(batch_id) = Kokkos::complex<double>(sum.real(), sum.imag());
            });
        });
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
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
Operator<internal::Prec, ExecutionSpace::Default>
Operator<internal::Prec, internal::Space>::copy_to_default_space() const {
    auto op =
        Operator<internal::Prec, ExecutionSpace::Default>::uninitialized_operator(_terms.extent(0));
    Kokkos::deep_copy(op._terms, _terms);
    op._is_view = false;
    op._is_hermitian = _is_hermitian;
    return op;
}

template <>
Operator<internal::Prec, ExecutionSpace::Host>
Operator<internal::Prec, internal::Space>::copy_to_host_space() const {
    auto op =
        Operator<internal::Prec, ExecutionSpace::Host>::uninitialized_operator(_terms.extent(0));
    Kokkos::deep_copy(op._terms, this->_terms);
    op._is_view = false;
    op._is_hermitian = _is_hermitian;
    return op;
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::calculate_default_mu() const {
    double mu = 0.;
    for (const auto& term : get_terms()) mu += std::abs(term.coef().real());
    return StdComplex(mu);
}

namespace {
template <Precision Prec, ExecutionSpace Space>
double validate_ground_state_input(const Operator<Prec, Space>& op,
                                   const StateVector<Prec, Space>& initial_state,
                                   std::uint64_t iter_count,
                                   std::optional<StdComplex> mu,
                                   std::optional<double> tolerance) {
    if (!op.n_terms() || !op.is_hermitian()) {
        throw std::runtime_error("Ground state solver requires a nonempty Hermitian operator.");
    }
    if (!iter_count) throw std::invalid_argument("Ground state solver requires iterations > 0.");
    if (mu && (!std::isfinite(mu->real()) || !std::isfinite(mu->imag()) || mu->imag() != 0.)) {
        throw std::invalid_argument("Ground state solver requires a finite real shift.");
    }
    if (tolerance && (!std::isfinite(*tolerance) || *tolerance <= 0.)) {
        throw std::invalid_argument("Ground state solver requires a positive finite tolerance.");
    }
    double bound = 0.;
    for (const auto& term : op.get_terms()) {
        const auto [x, z] = term.get_XZ_mask_representation();
        if (initial_state.n_qubits() < 64 && ((x | z) >> initial_state.n_qubits())) {
            throw std::invalid_argument("Operator acts on a qubit outside the initial state.");
        }
        const auto coef = term.coef();
        if (!std::isfinite(coef.real()) || !std::isfinite(coef.imag())) {
            throw std::invalid_argument("Ground state solver requires finite coefficients.");
        }
        bound += std::abs(coef.real());
    }
    const double norm = initial_state.get_squared_norm();
    if (!std::isfinite(bound) || !std::isfinite(norm) || norm <= 0.) {
        throw std::invalid_argument(
            "Ground state solver requires a finite nonzero initial state and operator.");
    }
    return bound;
}

template <Precision Prec>
constexpr double ground_state_default_tolerance() {
    if constexpr (Prec == Precision::F64) return 1e-10;
    if constexpr (Prec == Precision::F32) return 1e-5;
    return 0.05;
}

template <Precision Prec, ExecutionSpace Space>
typename Operator<Prec, Space>::GroundState ground_state_result(const Operator<Prec, Space>& op,
                                                                StateVector<Prec, Space> state,
                                                                std::uint64_t iterations,
                                                                double bound,
                                                                double tolerance) {
    const auto eigenvalue = op.get_expectation_value(state);
    auto residual = state.copy();
    op.apply_to_state(residual);
    residual.add_state_vector_with_coef(-eigenvalue, state);
    const double residual_norm = std::sqrt(residual.get_squared_norm());
    const bool converged =
        std::isfinite(residual_norm) && residual_norm <= tolerance * (bound > 0. ? bound : 1.);
    return {eigenvalue, state, residual_norm, iterations, converged};
}
}  // namespace

template <>
Operator<internal::Prec, internal::Space>::GroundState
Operator<internal::Prec, internal::Space>::solve_ground_state_by_power_method(
    const StateVector<internal::Prec, internal::Space>& initial_state,
    std::uint64_t iter_count,
    std::optional<StdComplex> mu,
    std::optional<double> tolerance) const {
    const double bound =
        validate_ground_state_input(*this, initial_state, iter_count, mu, tolerance);
    const double tol = tolerance.value_or(ground_state_default_tolerance<internal::Prec>());
    const StdComplex shift = mu.value_or(StdComplex(bound));
    if (!std::isfinite(shift.real()) || !std::isfinite(shift.imag())) {
        throw std::invalid_argument("Ground state solver requires a finite shift.");
    }
    auto state = initial_state.copy();
    state.normalize();
    for (std::uint64_t i = 0; i < iter_count; ++i) {
        auto result = ground_state_result(*this, state, i, bound, tol);
        if (result.converged) return result;
        auto next = state.copy();
        apply_to_state(next);
        next.add_state_vector_with_coef(-shift, state);
        const double norm = next.get_squared_norm();
        if (!std::isfinite(norm) || norm <= 0.) {
            throw std::runtime_error("Power method encountered a non-finite or zero iterate.");
        }
        next.normalize();
        state = next;
    }
    return ground_state_result(*this, state, iter_count, bound, tol);
}

template <>
Operator<internal::Prec, internal::Space>::GroundState
Operator<internal::Prec, internal::Space>::solve_ground_state_by_arnoldi_method(
    const StateVector<internal::Prec, internal::Space>& initial_state,
    std::uint64_t iter_count,
    std::optional<StdComplex> mu,
    std::optional<double> tolerance) const {
    const double bound =
        validate_ground_state_input(*this, initial_state, iter_count, mu, tolerance);
    const double tol = tolerance.value_or(ground_state_default_tolerance<internal::Prec>());
    const std::uint64_t count = std::min(iter_count, initial_state.dim());
    std::vector<StateVector<internal::Prec, internal::Space>> basis;
    basis.reserve(count);
    basis.push_back(initial_state.copy());
    basis.back().normalize();
    ComplexMatrix hessenberg = ComplexMatrix::Zero(count, count);
    std::uint64_t effective_count = count;
    constexpr double epsilon = [] {
        if constexpr (internal::Prec == Precision::F64)
            return std::numeric_limits<double>::epsilon();
        if constexpr (internal::Prec == Precision::F32)
            return static_cast<double>(std::numeric_limits<float>::epsilon());
        return 0.;
    }();
    for (std::uint64_t i = 0; i < count; ++i) {
        auto state = basis.back().copy();
        // A real shift does not change the Krylov space; omitting it avoids cancellation.
        apply_to_state(state);
        for (int pass = 0; pass < 2; ++pass) {
            for (std::uint64_t j = 0; j <= i; ++j) {
                const auto coef = internal::inner_product<internal::Prec, internal::Space>(
                    basis[j]._raw, state._raw);
                hessenberg(j, i) += static_cast<StdComplex>(coef);
                state.add_state_vector_with_coef(-coef, basis[j]);
            }
        }
        const double norm = std::sqrt(state.get_squared_norm());
        if (!std::isfinite(norm)) throw std::runtime_error("Non-finite Krylov norm.");
        if (i + 1 == count || norm <= 10. * epsilon * bound) {
            effective_count = i + 1;
            break;
        }
        hessenberg(i + 1, i) = norm;
        state.multiply_coef(1. / norm);
        basis.push_back(state);
    }
    Eigen::SelfAdjointEigenSolver<ComplexMatrix> solver(
        hessenberg.topLeftCorner(effective_count, effective_count));
    if (solver.info() != Eigen::ComputationInfo::Success)
        throw std::runtime_error("Krylov eigenvalue solver failed.");
    auto state =
        StateVector<internal::Prec, internal::Space>::uninitialized_state(initial_state.n_qubits());
    state.set_zero_norm_state();
    for (std::uint64_t i = 0; i < effective_count; ++i) {
        state.add_state_vector_with_coef(solver.eigenvectors()(i, 0), basis[i]);
    }
    state.normalize();
    return ground_state_result(*this, state, effective_count, bound, tol);
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
    ret._is_view = false;
    ret._is_hermitian = (nnz_count == 0);
    return ret;
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator*=(
    const PauliOperator<internal::Prec>& target) {
    Kokkos::parallel_for(
        "operator*=",
        Kokkos::RangePolicy<internal::SpaceType<internal::Space>>(0, _terms.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _terms(i) *= target; });
    _is_hermitian &= (target.coef().imag() == 0);
    return *this;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator*(
    const PauliOperator<internal::Prec>& target) const {
    return this->copy() *= target;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator+(
    const Operator<internal::Prec, internal::Space>& target) const {
    auto ret = Operator<internal::Prec, internal::Space>::uninitialized_operator(
        _terms.size() + target._terms.size());
    ret._is_view = false;
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
    const PauliOperator<internal::Prec>& target) const {
    auto ret = Operator<internal::Prec, internal::Space>::uninitialized_operator(_terms.size() + 1);
    ret._is_view = false;
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
