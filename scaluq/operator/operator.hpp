#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {

template <std::floating_point FloatType>
class Operator {
public:
    explicit Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

    [[nodiscard]] inline bool is_hermitian() { return _is_hermitian; }
    [[nodiscard]] inline std::uint64_t n_qubits() { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator<FloatType>>& terms() const {
        return _terms;
    }
    [[nodiscard]] std::string to_string() const {
        std::stringstream ss;
        for (auto itr = _terms.begin(); itr != _terms.end(); ++itr) {
            ss << itr->coef() << " " << itr->get_pauli_string();
            if (itr != prev(_terms.end())) {
                ss << " + ";
            }
        }
        return ss.str();
    }

    void add_operator(const PauliOperator<FloatType>& mpt) {
        add_operator(PauliOperator<FloatType>{mpt});
    }
    void add_operator(PauliOperator<FloatType>&& mpt) {
        _is_hermitian &= mpt.coef().imag() == 0.;
        if (![&] {
                const auto& target_list = mpt.target_qubit_list();
                if (target_list.empty()) return true;
                return *std::max_element(target_list.begin(), target_list.end()) < _n_qubits;
            }()) {
            throw std::runtime_error(
                "Operator::add_operator: target index of pauli_operator is larger than "
                "n_qubits");
        }
        this->_terms.emplace_back(std::move(mpt));
    }

    void add_random_operator(const std::uint64_t operator_count = 1,
                             std::uint64_t seed = std::random_device()()) {
        Random random(seed);
        for (std::uint64_t operator_idx = 0; operator_idx < operator_count; operator_idx++) {
            std::vector<std::uint64_t> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
            for (std::uint64_t qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
                target_qubit_list[qubit_idx] = qubit_idx;
                pauli_id_list[qubit_idx] = random.int32() & 0b11;
            }
            Complex coef = random.uniform() * 2. - 1.;
            this->add_operator(PauliOperator(target_qubit_list, pauli_id_list, coef));
        }
    }

    void optimize() {
        std::map<std::tuple<std::uint64_t, std::uint64_t>, Complex> pauli_and_coef;
        for (const auto& pauli : _terms) {
            pauli_and_coef[pauli.get_XZ_mask_representation()] += pauli.coef();
        }
        _terms.clear();
        for (const auto& [mask, coef] : pauli_and_coef) {
            const auto& [x_mask, z_mask] = mask;
            _terms.emplace_back(x_mask, z_mask, coef);
        }
    }

    [[nodiscard]] Operator get_dagger() const {
        Operator quantum_operator(_n_qubits);
        for (const auto& pauli : _terms) {
            quantum_operator.add_operator(pauli.get_dagger());
        }
        return quantum_operator;
    }

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector<FloatType>& state_vector) const {
        StateVector<FloatType> res(state_vector.n_qubits());
        res.set_zero_norm_state();
        for (const auto& term : _terms) {
            StateVector<double> tmp = state_vector.copy();
            term.apply_to_state(tmp);
            res.add_state_vector_with_coef(1, tmp);
        }
        state_vector = res;
    }

    [[nodiscard]] Complex get_expectation_value(const StateVector<FloatType>& state_vector) const {
        if (_n_qubits > state_vector.n_qubits()) {
            throw std::runtime_error(
                "Operator::get_expectation_value: n_qubits of state_vector is too small");
        }
        std::uint64_t nterms = _terms.size();
        Kokkos::View<const PauliOperator<FloatType>*,
                     Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            terms_view(_terms.data(), nterms);
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> bmasks_host("bmasks_host", nterms);
        Kokkos::View<std::uint64_t*, Kokkos::HostSpace> pmasks_host("pmasks_host", nterms);
        Kokkos::View<Complex*, Kokkos::HostSpace> coefs_host("coefs_host", nterms);
        Kokkos::Experimental::transform(
            Kokkos::DefaultHostExecutionSpace(),
            terms_view,
            bmasks_host,
            [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_bit_flip_mask; });
        Kokkos::Experimental::transform(
            Kokkos::DefaultHostExecutionSpace(),
            terms_view,
            pmasks_host,
            [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_phase_flip_mask; });
        Kokkos::Experimental::transform(
            Kokkos::DefaultHostExecutionSpace(),
            terms_view,
            coefs_host,
            [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_coef; });
        Kokkos::View<std::uint64_t*> bmasks("bmasks", nterms);
        Kokkos::View<std::uint64_t*> pmasks("pmasks", nterms);
        Kokkos::View<Complex*> coefs("coefs", nterms);
        Kokkos::deep_copy(bmasks, bmasks_host);
        Kokkos::deep_copy(pmasks, pmasks_host);
        Kokkos::deep_copy(coefs, coefs_host);
        std::uint64_t dim = state_vector.dim();
        Complex res;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nterms, dim >> 1}),
            KOKKOS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, Complex & res_lcl) {
                std::uint64_t bit_flip_mask = bmasks[term_id];
                std::uint64_t phase_flip_mask = pmasks[term_id];
                Complex coef = coefs[term_id];
                if (bit_flip_mask == 0) {
                    std::uint64_t state_idx1 = state_idx << 1;
                    double tmp1 = (Kokkos::conj(state_vector._raw[state_idx1]) *
                                   state_vector._raw[state_idx1])
                                      .real();
                    if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                    std::uint64_t state_idx2 = state_idx1 | 1;
                    double tmp2 = (Kokkos::conj(state_vector._raw[state_idx2]) *
                                   state_vector._raw[state_idx2])
                                      .real();
                    if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                    res_lcl += coef * (tmp1 + tmp2);
                } else {
                    std::uint64_t pivot =
                        sizeof(std::uint64_t) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                    std::uint64_t global_phase_90rot_count =
                        Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                    Complex global_phase = internal::PHASE_90ROT()[global_phase_90rot_count % 4];
                    std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                    std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                    double tmp =
                        Kokkos::real(state_vector._raw[basis_0] *
                                     Kokkos::conj(state_vector._raw[basis_1]) * global_phase * 2.);
                    if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                    res_lcl += coef * tmp;
                }
            },
            res);
        Kokkos::fence();
        return res;
    }

    [[nodiscard]] Complex get_transition_amplitude(
        const StateVector<FloatType>& state_vector_bra,
        const StateVector<FloatType>& state_vector_ket) const {
        if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
            throw std::runtime_error(
                "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
                "state_vector_ket must be same");
        }
        if (_n_qubits > state_vector_bra.n_qubits()) {
            throw std::runtime_error(
                "Operator::get_transition_amplitude: n_qubits of state_vector is too "
                "small");
        }
        std::uint64_t nterms = _terms.size();
        std::vector<std::uint64_t> bmasks_vector(nterms);
        std::vector<std::uint64_t> pmasks_vector(nterms);
        std::vector<Complex> coefs_vector(nterms);
        std::transform(
            _terms.begin(),
            _terms.end(),
            bmasks_vector.begin(),
            [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_bit_flip_mask; });
        std::transform(
            _terms.begin(),
            _terms.end(),
            pmasks_vector.begin(),
            [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_phase_flip_mask; });
        std::transform(_terms.begin(),
                       _terms.end(),
                       coefs_vector.begin(),
                       [](const PauliOperator<FloatType>& pauli) { return pauli._ptr->_coef; });
        Kokkos::View<std::uint64_t*> bmasks =
            internal::convert_host_vector_to_device_view(bmasks_vector);
        Kokkos::View<std::uint64_t*> pmasks =
            internal::convert_host_vector_to_device_view(pmasks_vector);
        Kokkos::View<Complex*> coefs = internal::convert_host_vector_to_device_view(coefs_vector);
        std::uint64_t dim = state_vector_bra.dim();
        Complex res;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nterms, dim >> 1}),
            KOKKOS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, Complex & res_lcl) {
                std::uint64_t bit_flip_mask = bmasks[term_id];
                std::uint64_t phase_flip_mask = pmasks[term_id];
                Complex coef = coefs[term_id];
                if (bit_flip_mask == 0) {
                    std::uint64_t state_idx1 = state_idx << 1;
                    Complex tmp1 = (Kokkos::conj(state_vector_bra._raw[state_idx1]) *
                                    state_vector_ket._raw[state_idx1]);
                    if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                    std::uint64_t state_idx2 = state_idx1 | 1;
                    Complex tmp2 = (Kokkos::conj(state_vector_bra._raw[state_idx2]) *
                                    state_vector_ket._raw[state_idx2]);
                    if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                    res_lcl += coef * (tmp1 + tmp2);
                } else {
                    std::uint64_t pivot =
                        sizeof(std::uint64_t) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                    std::uint64_t global_phase_90rot_count =
                        Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                    Complex global_phase = internal::PHASE_90ROT()[global_phase_90rot_count % 4];
                    std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                    std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                    Complex tmp1 = Kokkos::conj(state_vector_bra._raw[basis_1]) *
                                   state_vector_ket._raw[basis_0] * global_phase;
                    if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                    Complex tmp2 = Kokkos::conj(state_vector_bra._raw[basis_0]) *
                                   state_vector_ket._raw[basis_1] * global_phase;
                    if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                    res_lcl += coef * (tmp1 + tmp2);
                }
            },
            res);
        Kokkos::fence();
        return res;
    }

    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_arnoldi_method(
        const StateVector<FloatType>& state, std::uint64_t iter_count, Complex mu = 0.) const;
    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_power_method(
        const StateVector<FloatType>& state, std::uint64_t iter_count, Complex mu = 0.) const;

    Operator& operator*=(Complex coef) {
        for (auto& pauli : _terms) {
            pauli = pauli * coef;
        }
        return *this;
    }
    Operator operator*(Complex coef) const { return Operator(*this) *= coef; }
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1; }
    Operator& operator+=(const Operator& target) {
        if (_n_qubits != target._n_qubits) {
            throw std::runtime_error("Operator::oeprator+=: n_qubits must be equal");
        }
        for (const auto& pauli : target._terms) {
            add_operator(pauli);
        }
        return *this;
    }
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const {
        if (_n_qubits != target._n_qubits) {
            throw std::runtime_error("Operator::oeprator+=: n_qubits must be equal");
        }
        Operator ret(_n_qubits);
        for (const auto& pauli1 : _terms) {
            for (const auto& pauli2 : target._terms) {
                ret.add_operator(pauli1 * pauli2);
            }
        }
        return ret;
    }
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator<FloatType>& pauli) {
        add_operator(pauli);
        return *this;
    }
    Operator operator+(const PauliOperator<FloatType>& pauli) const {
        return Operator(*this) += pauli;
    }
    Operator& operator-=(const PauliOperator<FloatType>& pauli) { return *this += pauli * -1; }
    Operator operator-(const PauliOperator<FloatType>& pauli) const {
        return Operator(*this) -= pauli;
    }
    Operator& operator*=(const PauliOperator<FloatType>& pauli) {
        for (auto& pauli1 : _terms) {
            pauli1 = pauli1 * pauli;
        }
        return *this;
    }
    Operator operator*(const PauliOperator<FloatType>& pauli) const {
        return Operator(*this) *= pauli;
    }

private:
    std::vector<PauliOperator<FloatType>> _terms;
    std::uint64_t _n_qubits;
    bool _is_hermitian = true;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_operator_operator_hpp(nb::module_& m) {
    nb::class_<Operator>(m, "Operator", "General quantum operator class.")
        .def(nb::init<std::uint64_t>(),
             "qubit_count"_a,
             "Initialize operator with specified number of qubits.")
        .def("is_hermitian", &Operator::is_hermitian, "Check if the operator is Hermitian.")
        .def("n_qubits", &Operator::n_qubits, "Get the number of qubits the operator acts on.")
        .def("terms", &Operator::terms, "Get the list of Pauli terms that make up the operator.")
        .def("to_string", &Operator::to_string, "Get string representation of the operator.")
        .def("add_operator",
             nb::overload_cast<const PauliOperator&>(&Operator::add_operator),
             "Add a Pauli operator to this operator.")
        .def(
            "add_random_operator",
            [](Operator& op, std::uint64_t operator_count, std::optional<std::uint64_t> seed) {
                return op.add_random_operator(operator_count,
                                              seed.value_or(std::random_device{}()));
            },
            "operator_count"_a,
            "seed"_a = std::nullopt,
            "Add a specified number of random Pauli operators to this operator. An optional seed "
            "can be provided for reproducibility.")
        .def("optimize", &Operator::optimize, "Optimize the operator by combining like terms.")
        .def("get_dagger",
             &Operator::get_dagger,
             "Get the adjoint (Hermitian conjugate) of the operator.")
        .def("apply_to_state", &Operator::apply_to_state, "Apply the operator to a state vector.")
        .def("get_expectation_value",
             &Operator::get_expectation_value,
             "Get the expectation value of the operator with respect to a state vector.")
        .def("get_transition_amplitude",
             &Operator::get_transition_amplitude,
             "Get the transition amplitude of the operator between two state vectors.")
        .def(nb::self *= Complex())
        .def(nb::self * Complex())
        .def(+nb::self)
        .def(-nb::self)
        .def(nb::self += nb::self)
        .def(nb::self + nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self += PauliOperator())
        .def(nb::self + PauliOperator())
        .def(nb::self -= PauliOperator())
        .def(nb::self - PauliOperator())
        .def(nb::self *= PauliOperator())
        .def(nb::self * PauliOperator());
}
}  // namespace internal
#endif
}  // namespace scaluq
