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
    explicit OperatorBatched(const std::vector<Operator<Prec, Space>>& ops)
        : _row_ptr("operator_row_ptr", ops.size() + 1) {
        std::vector<std::uint64_t> row_ptr_h;
        row_ptr_h.push_back(0);
        for (const auto& op : ops) {
            row_ptr_h.push_back(row_ptr_h.back() + op.n_terms());
        }
        auto row_ptr_h_view = internal::wrapped_host_view(row_ptr_h);
        Kokkos::deep_copy(_row_ptr, row_ptr_h_view);

        _ops = Kokkos::View<Pauli*, ExecutionSpaceType>("operator_ops", row_ptr_h.back());
        for (std::uint64_t i = 0; i < ops.size(); ++i) {
            auto terms = ops[i].get_terms();  // TODO: _terms を直に取りたい
            auto terms_h_view = internal::wrapped_host_view(terms);
            Kokkos::deep_copy(
                Kokkos::subview(_ops, Kokkos::make_pair(row_ptr_h[i], row_ptr_h[i + 1])),
                terms_h_view);
        }
    }

    [[nodiscard]] OperatorBatched copy() const;
    void load(const std::vector<PauliOperator<Prec, Space>>& terms);

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::uint64_t batch_size() const { return _row_ptr.extent(0) - 1; }

    [[nodiscard]] OperatorBatched get_dagger() const;

    StateVectorBatched<Prec, Space> get_applied_to_states(
        const StateVector<Prec, Space>& state_vector) const;

    [[nodiscard]] std::vector<StdComplex> get_expectation_value(
        const StateVector<Prec, Space>& state_vector) const;

    [[nodiscard]] std::vector<StdComplex> get_transition_amplitude(
        const StateVector<Prec, Space>& state_vector_bra,
        const StateVector<Prec, Space>& state_vector_ket) const;

    // // not implemented yet
    [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_arnoldi_method(
        const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;
    // // not implemented yet
    [[nodiscard]] StdComplex solve_ground_state_eigenvalue_by_power_method(
        const StateVector<Prec, Space>& state, std::uint64_t iter_count, StdComplex mu = 0.) const;

    [[nodiscard]] Operator<Prec, Space> get_operator_at(std::uint64_t index) const;
    [[nodiscard]] std::vector<Operator<Prec, Space>> get_operators() const;

    OperatorBatched operator*(const std::vector<StdComplex>& coef) const;
    OperatorBatched& operator*=(const std::vector<StdComplex>& coef);
    OperatorBatched operator+() const { return *this; }
    OperatorBatched operator-() const { return *this * -1.; }
    OperatorBatched operator+(const OperatorBatched& target) const;
    OperatorBatched operator-(const OperatorBatched& target) const { return *this + target * -1.; }
    OperatorBatched operator*(const OperatorBatched& target) const;
    OperatorBatched operator+(const std::vector<PauliOperator<Prec, Space>>& pauli) const;
    OperatorBatched operator-(const std::vector<PauliOperator<Prec, Space>>& pauli) const {
        return *this + pauli * -1.;
    }
    OperatorBatched operator*(const std::vector<PauliOperator<Prec, Space>>& pauli) const;
    OperatorBatched& operator*=(const std::vector<PauliOperator<Prec, Space>>& pauli);

    friend void to_json(Json& j, const OperatorBatched& op) {
        j.clear();
        j["Operators"] = Json::array();
        for (const auto& pauli : op.get_operators()) {
            Json tmp = pauli;
            j["Operators"].push_back(tmp);
        }
    }
    friend void from_json(const Json& j, OperatorBatched& op) {
        std::vector<Operator<Prec, Space>> res;
        for (const auto& item : j.at("Operators")) {
            std::vector<PauliOperator<Prec, Space>> terms;
            for (const auto& term : item.at("terms")) {
                std::string pauli_string = term.at("pauli_string").get<std::string>();
                StdComplex coef = term.at("coef").get<StdComplex>();
                terms.emplace_back(pauli_string, coef);
            }
            res.emplace_back(terms);
        }
        op = OperatorBatched<Prec, Space>(res);
    }

    friend std::ostream& operator<<(std::ostream& os, const OperatorBatched& op) {
        return os << op.to_string();
    }

private:
    Kokkos::View<PauliOperator<Prec, Space>*, ExecutionSpaceType> _ops;
    Kokkos::View<std::uint64_t*, ExecutionSpaceType> _row_ptr;
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_operator_operator_batched_hpp(nb::module_& m) {
    nb::class_<OperatorBatched<Prec, Space>>(
        m, "Operator", DocString().desc("General quantum operator class for batched operators."))
        .def(nb::init<>(), DocString().desc("Default constructor."))
        .def(nb::init<const std::vector<std::vector<PauliOperator<Prec, Space>>>&>(),
             DocString().desc("Constructor from a vector of Pauli operators."))
        .def(nb::init<const std::vector<Operator<Prec, Space>>&>(),
             DocString().desc("Constructor from a vector of Operators."))
        .def("copy", &OperatorBatched<Prec, Space>::copy, DocString().desc("Return a copy."))
        .def("load",
             &OperatorBatched<Prec, Space>::load,
             DocString()
                 .desc("Load a vector of Pauli operators.")
                 .arg("terms", "A vector of Pauli operators."))
        .def("to_string",
             &OperatorBatched<Prec, Space>::to_string,
             DocString().desc("Return a string representation of the operator."))
        .def("batch_size",
             &OperatorBatched<Prec, Space>::batch_size,
             DocString().desc("Return the batch size of the operator."))
        .def("get_dagger",
             &OperatorBatched<Prec, Space>::get_dagger,
             DocString().desc("Return the Hermitian conjugate of the operator."))
        .def("get_applied_to_states",
             &OperatorBatched<Prec, Space>::get_applied_to_states,
             DocString()
                 .desc("Apply the batched operator to a state vector.")
                 .arg("state_vector", "A state vector to be applied."))
        .def("get_expectation_value",
             &OperatorBatched<Prec, Space>::get_expectation_value,
             DocString()
                 .desc("Return a vector of expectation values for each operator.")
                 .arg("state_vector", "A state vector to compute expectation values."))
        .def("get_transition_amplitude",
             &OperatorBatched<Prec, Space>::get_transition_amplitude,
             DocString()
                 .desc("Return a vector of transition amplitudes for each operator.")
                 .arg("state_vector_bra", "A bra state vector.")
                 .arg("state_vector_ket", "A ket state vector."))
        .def("solve_ground_state_eigenvalue_by_arnoldi_method",
             &OperatorBatched<Prec, Space>::solve_ground_state_eigenvalue_by_arnoldi_method,
             DocString()
                 .desc("Solve the ground state eigenvalue using the Arnoldi method.")
                 .arg("state", "An initial state vector.")
                 .arg("iter_count", "Number of iterations.")
                 .arg("mu", "A shift parameter (default is 0)."))
        .def("solve_ground_state_eigenvalue_by_power_method",
             &OperatorBatched<Prec, Space>::solve_ground_state_eigenvalue_by_power_method,
             DocString()
                 .desc("Solve the ground state eigenvalue using the Power method.")
                 .arg("state", "An initial state vector.")
                 .arg("iter_count", "Number of iterations.")
                 .arg("mu", "A shift parameter (default is 0)."))
        .def("get_operator_at",
             &OperatorBatched<Prec, Space>::get_operator_at,
             DocString()
                 .desc("Return the operator at the specified index.")
                 .arg("index", "Index of the operator."))
        .def("get_operators",
             &OperatorBatched<Prec, Space>::get_operators,
             DocString().desc("Return a vector of all operators."))
        .def(nb::self * std::vector<StdComplex>())
        .def(nb::self *= std::vector<StdComplex>())
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self + std::vector<PauliOperator<Prec, Space>>())
        .def(nb::self - std::vector<PauliOperator<Prec, Space>>())
        .def(nb::self * std::vector<PauliOperator<Prec, Space>>())
        .def(nb::self *= std::vector<PauliOperator<Prec, Space>>());
}
}  // namespace internal
#endif

}  // namespace scaluq
