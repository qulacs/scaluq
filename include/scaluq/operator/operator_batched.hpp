#pragma once

#include <random>
#include <vector>

#include "operator.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class OperatorBatched {
    using ComplexType = internal::Complex<Prec>;
    using FloatType = internal::Float<Prec>;
    using ExecutionSpaceType = internal::SpaceType<Space>;
    using Pauli = PauliOperator<Prec, Space>;

public:
    OperatorBatched() = default;
    explicit OperatorBatched(const std::vector<std::vector<Pauli>>& ops)
        : _row_ptr("operator_row_ptr", ops.size() + 1) {
        std::vector<std::uint64_t> row_ptr_h;
        row_ptr_h.push_back(0);
        for (const auto& op : ops) {
            row_ptr_h.push_back(row_ptr_h.back() + op.size());
        }
        auto row_ptr_h_view = internal::wrapped_host_view(row_ptr_h);
        Kokkos::deep_copy(_row_ptr, row_ptr_h_view);

        _ops = Kokkos::View<Pauli*, ExecutionSpaceType>("operator_ops", row_ptr_h.back());
        std::vector<Pauli> ops_h;
        for (const auto& op : ops) {
            ops_h.insert(ops_h.end(), op.begin(), op.end());
        }
        auto ops_h_view = internal::wrapped_host_view(ops_h);
        Kokkos::deep_copy(_ops, ops_h_view);
    }

    [[nodiscard]] OperatorBatched copy() const;
    // void load(const std::vector<PauliOperator<Prec, Space>>& terms);
    // static Operator uninitialized_operator(std::uint64_t n_terms);

    // [[nodiscard]] inline std::vector<PauliOperator<Prec, Space>> get_terms() const {
    //     return internal::convert_view_to_vector<PauliOperator<Prec, Space>, Space>(_terms);
    // }
    // [[nodiscard]] std::string to_string() const;

    // void optimize();

    // [[nodiscard]] Operator get_dagger() const;

    // [[nodiscard]] ComplexMatrix get_full_matrix(std::uint64_t n_qubits) const;

    // void apply_to_state(StateVector<Prec, Space>& state_vector) const;

    [[nodiscard]] std::vector<StdComplex> get_expectation_value(
        const StateVector<Prec, Space>& state_vector) const;
    // [[nodiscard]] std::vector<StdComplex> get_expectation_value(
    //     const StateVectorBatched<Prec, Space>& states) const;

    // [[nodiscard]] ComplexMatrix get_matrix_ignoring_coef() const;

    // [[nodiscard]] StdComplex get_transition_amplitude(
    //     const StateVector<Prec, Space>& state_vector_bra,
    //     const StateVector<Prec, Space>& state_vector_ket) const;
    // [[nodiscard]] std::vector<StdComplex> get_transition_amplitude(
    //     const StateVectorBatched<Prec, Space>& states_bra,
    //     const StateVectorBatched<Prec, Space>& states_ket) const;

    // // not implemented yet
    // [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_arnoldi_method(
    //     const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.)
    //     const;
    // // not implemented yet
    // [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_power_method(
    //     const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.)
    //     const;

    // Operator operator*(StdComplex coef) const;
    // Operator& operator*=(StdComplex coef);
    // Operator operator+() const { return *this; }
    // Operator operator-() const { return *this * -1.; }
    // Operator operator+(const Operator& target) const;
    // Operator operator-(const Operator& target) const { return *this + target * -1.; }
    // Operator operator*(const Operator& target) const;
    // Operator operator+(const PauliOperator<Prec, Space>& pauli) const;
    // Operator operator-(const PauliOperator<Prec, Space>& pauli) const {
    //     return *this + pauli * -1.;
    // }
    // Operator operator*(const PauliOperator<Prec, Space>& pauli) const;
    // Operator& operator*=(const PauliOperator<Prec, Space>& pauli);

    // friend void to_json(Json& j, const Operator& op) {
    //     j.clear();
    //     j["terms"] = Json::array();
    //     for (const auto& pauli : op.get_terms()) {
    //         Json tmp = pauli;
    //         j["terms"].push_back(tmp);
    //     }
    // }
    // friend void from_json(const Json& j, Operator& op) {
    //     std::vector<PauliOperator<Prec, Space>> res;
    //     for (const auto& term : j.at("terms")) {
    //         std::string pauli_string = term.at("pauli_string").get<std::string>();
    //         StdComplex coef = term.at("coef").get<StdComplex>();
    //         res.emplace_back(pauli_string, coef);
    //     }
    //     op = Operator<Prec, Space>(res);
    // }

    // friend std::ostream& operator<<(std::ostream& os, const Operator& op) {
    //     return os << op.to_string();
    // }

private:
    Kokkos::View<PauliOperator<Prec, Space>*, ExecutionSpaceType> _ops;
    Kokkos::View<std::uint64_t*, ExecutionSpaceType> _row_ptr;
};

}  // namespace scaluq
