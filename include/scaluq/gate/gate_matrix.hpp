#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <ranges>
#include <vector>

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
template <std::floating_point Fp>
class DenseMatrixGateImpl : public GateBase<Fp> {
    Matrix<Fp> _matrix;
    bool _is_unitary;

public:
    DenseMatrixGateImpl(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const ComplexMatrix<Fp>& mat,
                        bool is_unitary = false);

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;

    Matrix<Fp> get_matrix_internal() const;

    ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "DensetMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

template <std::floating_point Fp>
class SparseMatrixGateImpl : public GateBase<Fp> {
    SparseMatrix<Fp> _matrix;
    std::uint64_t num_nnz;

public:
    SparseMatrixGateImpl(std::uint64_t target_mask,
                         std::uint64_t control_mask,
                         const SparseComplexMatrix<Fp>& mat);

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;

    Matrix<Fp> get_matrix_internal() const;

    ComplexMatrix<Fp> get_matrix() const override;

    SparseComplexMatrix<Fp> get_sparse_matrix() const { return get_matrix().sparseView(); }

    void update_quantum_state(StateVector<Fp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SparseMatrix"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"matrix", "Not inplemented yet"}};
    }
};

}  // namespace internal

template <std::floating_point Fp>
using SparseMatrixGate = internal::GatePtr<internal::SparseMatrixGateImpl<Fp>>;
template <std::floating_point Fp>
using DenseMatrixGate = internal::GatePtr<internal::DenseMatrixGateImpl<Fp>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_ONETARGETMATRIXGATE_WITH_TYPE(Type)                              \
    template <>                                                                                \
    inline std::shared_ptr<const OneTargetMatrixGateImpl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                       \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                     \
        auto matrix = j.at("matrix").get<std::vector<std::vector<Kokkos::complex<Type>>>>();   \
        return std::make_shared<const OneTargetMatrixGateImpl<Type>>(                          \
            vector_to_mask(targets),                                                           \
            vector_to_mask(controls),                                                          \
            std::array<std::array<Kokkos::complex<Type>, 2>, 2>{                               \
                matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]});                      \
    }
DECLARE_GET_FROM_JSON_ONETARGETMATRIXGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_ONETARGETMATRIXGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_ONETARGETMATRIXGATE_WITH_TYPE

#define DECLARE_GET_FROM_JSON_TWOTARGETMATRIXGATE_WITH_TYPE(Type)                              \
    template <>                                                                                \
    inline std::shared_ptr<const TwoTargetMatrixGateImpl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                       \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                     \
        auto matrix = j.at("matrix").get<std::vector<std::vector<Kokkos::complex<Type>>>>();   \
        return std::make_shared<const TwoTargetMatrixGateImpl<Type>>(                          \
            vector_to_mask(targets),                                                           \
            vector_to_mask(controls),                                                          \
            std::array<std::array<Kokkos::complex<Type>, 4>, 4>{matrix[0][0],                  \
                                                                matrix[0][1],                  \
                                                                matrix[0][2],                  \
                                                                matrix[0][3],                  \
                                                                matrix[1][0],                  \
                                                                matrix[1][1],                  \
                                                                matrix[1][2],                  \
                                                                matrix[1][3],                  \
                                                                matrix[2][0],                  \
                                                                matrix[2][1],                  \
                                                                matrix[2][2],                  \
                                                                matrix[2][3],                  \
                                                                matrix[3][0],                  \
                                                                matrix[3][1],                  \
                                                                matrix[3][2],                  \
                                                                matrix[3][3]});                \
    }
DECLARE_GET_FROM_JSON_TWOTARGETMATRIXGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_TWOTARGETMATRIXGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_TWOTARGETMATRIXGATE_WITH_TYPE

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_gate_matrix_hpp(nb::module_& m) {
    DEF_GATE(SparseMatrixGate, Fp, "Specific class of sparse matrix gate.")
        .def("matrix", [](const SparseMatrixGate<Fp>& gate) { return gate->get_matrix(); })
        .def("sparse_matrix",
             [](const SparseMatrixGate<Fp>& gate) { return gate->get_sparse_matrix(); });
    DEF_GATE(DenseMatrixGate, Fp, "Specific class of dense matrix gate.")
        .def("matrix", [](const DenseMatrixGate<Fp>& gate) { return gate->get_matrix(); });
}
}  // namespace internal
#endif
}  // namespace scaluq
