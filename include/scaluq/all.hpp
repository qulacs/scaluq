#pragma once

#include "circuit/circuit.hpp"
#include "constant.hpp"
#include "gate/gate.hpp"
#include "gate/gate_factory.hpp"
#include "gate/merge_gate.hpp"
#include "gate/param_gate.hpp"
#include "gate/param_gate_factory.hpp"
#include "kokkos.hpp"
#include "operator/operator.hpp"
#include "operator/pauli_operator.hpp"
#include "state/state_vector.hpp"
#include "state/state_vector_batched.hpp"
#include "types.hpp"
#include "util/random.hpp"

/*
 * Calling this macro from your original namespace will create aliases for all the types and
 * functions without template arguments.
 */
#define SCALUQ_OMIT_TEMPLATE(Prec, Space)                                                          \
    using StateVector = ::scaluq::StateVector<Prec, Space>;                                        \
    using StateVectorBatched = ::scaluq::StateVectorBatched<Prec, Space>;                          \
    using PauliOperator = ::scaluq::PauliOperator<Prec, Space>;                                    \
    using Operator = ::scaluq::Operator<Prec, Space>;                                              \
    using Gate = ::scaluq::Gate<Prec, Space>;                                                      \
    using IGate = ::scaluq::IGate<Prec, Space>;                                                    \
    using GlobalPhaseGate = ::scaluq::GlobalPhaseGate<Prec, Space>;                                \
    using XGate = ::scaluq::XGate<Prec, Space>;                                                    \
    using YGate = ::scaluq::YGate<Prec, Space>;                                                    \
    using ZGate = ::scaluq::ZGate<Prec, Space>;                                                    \
    using HGate = ::scaluq::HGate<Prec, Space>;                                                    \
    using SGate = ::scaluq::SGate<Prec, Space>;                                                    \
    using SdagGate = ::scaluq::SdagGate<Prec, Space>;                                              \
    using TGate = ::scaluq::TGate<Prec, Space>;                                                    \
    using TdagGate = ::scaluq::TdagGate<Prec, Space>;                                              \
    using SqrtXGate = ::scaluq::SqrtXGate<Prec, Space>;                                            \
    using SqrtXdagGate = ::scaluq::SqrtXdagGate<Prec, Space>;                                      \
    using SqrtYGate = ::scaluq::SqrtYGate<Prec, Space>;                                            \
    using SqrtYdagGate = ::scaluq::SqrtYdagGate<Prec, Space>;                                      \
    using P0Gate = ::scaluq::P0Gate<Prec, Space>;                                                  \
    using P1Gate = ::scaluq::P1Gate<Prec, Space>;                                                  \
    using RXGate = ::scaluq::RXGate<Prec, Space>;                                                  \
    using RYGate = ::scaluq::RYGate<Prec, Space>;                                                  \
    using RZGate = ::scaluq::RZGate<Prec, Space>;                                                  \
    using U1Gate = ::scaluq::U1Gate<Prec, Space>;                                                  \
    using U2Gate = ::scaluq::U2Gate<Prec, Space>;                                                  \
    using U3Gate = ::scaluq::U3Gate<Prec, Space>;                                                  \
    using SwapGate = ::scaluq::SwapGate<Prec, Space>;                                              \
    using PauliGate = ::scaluq::PauliGate<Prec, Space>;                                            \
    using PauliRotationGate = ::scaluq::PauliRotationGate<Prec, Space>;                            \
    using SparseMatrixGate = ::scaluq::SparseMatrixGate<Prec, Space>;                              \
    using DenseMatrixGate = ::scaluq::DenseMatrixGate<Prec, Space>;                                \
    using ProbabilisticGate = ::scaluq::ProbabilisticGate<Prec, Space>;                            \
    namespace gate {                                                                               \
    inline auto& I = ::scaluq::gate::I<Prec, Space>;                                               \
    inline Gate GlobalPhase(double phase,                                                          \
                            const std::vector<std::uint64_t>& controls = {},                       \
                            std::vector<std::uint64_t> control_values = {}) {                      \
        return ::scaluq::gate::GlobalPhase<Prec, Space>(phase, controls, control_values);          \
    }                                                                                              \
    inline Gate X(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::X<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate Y(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::Y<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate Z(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::Z<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate H(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::H<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate S(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::S<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate Sdag(std::uint64_t target,                                                         \
                     const std::vector<std::uint64_t>& controls = {},                              \
                     std::vector<std::uint64_t> control_values = {}) {                             \
        return ::scaluq::gate::Sdag<Prec, Space>(target, controls, control_values);                \
    }                                                                                              \
    inline Gate T(std::uint64_t target,                                                            \
                  const std::vector<std::uint64_t>& controls = {},                                 \
                  std::vector<std::uint64_t> control_values = {}) {                                \
        return ::scaluq::gate::T<Prec, Space>(target, controls, control_values);                   \
    }                                                                                              \
    inline Gate Tdag(std::uint64_t target,                                                         \
                     const std::vector<std::uint64_t>& controls = {},                              \
                     std::vector<std::uint64_t> control_values = {}) {                             \
        return ::scaluq::gate::Tdag<Prec, Space>(target, controls, control_values);                \
    }                                                                                              \
    inline Gate SqrtX(std::uint64_t target,                                                        \
                      const std::vector<std::uint64_t>& controls = {},                             \
                      std::vector<std::uint64_t> control_values = {}) {                            \
        return ::scaluq::gate::SqrtX<Prec, Space>(target, controls, control_values);               \
    }                                                                                              \
    inline Gate SqrtXdag(std::uint64_t target,                                                     \
                         const std::vector<std::uint64_t>& controls = {},                          \
                         std::vector<std::uint64_t> control_values = {}) {                         \
        return ::scaluq::gate::SqrtXdag<Prec, Space>(target, controls, control_values);            \
    }                                                                                              \
    inline Gate SqrtY(std::uint64_t target,                                                        \
                      const std::vector<std::uint64_t>& controls = {},                             \
                      std::vector<std::uint64_t> control_values = {}) {                            \
        return ::scaluq::gate::SqrtY<Prec, Space>(target, controls, control_values);               \
    }                                                                                              \
    inline Gate SqrtYdag(std::uint64_t target,                                                     \
                         const std::vector<std::uint64_t>& controls = {},                          \
                         std::vector<std::uint64_t> control_values = {}) {                         \
        return ::scaluq::gate::SqrtYdag<Prec, Space>(target, controls, control_values);            \
    }                                                                                              \
    inline Gate P0(std::uint64_t target,                                                           \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::P0<Prec, Space>(target, controls, control_values);                  \
    }                                                                                              \
    inline Gate P1(std::uint64_t target,                                                           \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::P1<Prec, Space>(target, controls, control_values);                  \
    }                                                                                              \
    inline Gate RX(std::uint64_t target,                                                           \
                   double angle,                                                                   \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::RX<Prec, Space>(target, angle, controls, control_values);           \
    }                                                                                              \
    inline Gate RY(std::uint64_t target,                                                           \
                   double angle,                                                                   \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::RY<Prec, Space>(target, angle, controls, control_values);           \
    }                                                                                              \
    inline Gate RZ(std::uint64_t target,                                                           \
                   double angle,                                                                   \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::RZ<Prec, Space>(target, angle, controls, control_values);           \
    }                                                                                              \
    inline Gate U1(std::uint64_t target,                                                           \
                   double lambda,                                                                  \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::U1<Prec, Space>(target, lambda, controls, control_values);          \
    }                                                                                              \
    inline Gate U2(std::uint64_t target,                                                           \
                   double phi,                                                                     \
                   double lambda,                                                                  \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::U2<Prec, Space>(target, phi, lambda, controls, control_values);     \
    }                                                                                              \
    inline Gate U3(std::uint64_t target,                                                           \
                   double theta,                                                                   \
                   double phi,                                                                     \
                   double lambda,                                                                  \
                   const std::vector<std::uint64_t>& controls = {},                                \
                   std::vector<std::uint64_t> control_values = {}) {                               \
        return ::scaluq::gate::U3<Prec, Space>(                                                    \
            target, theta, phi, lambda, controls, control_values);                                 \
    }                                                                                              \
    inline auto& CX = ::scaluq::gate::CX<Prec, Space>;                                             \
    inline auto& CNot = ::scaluq::gate::CNot<Prec, Space>;                                         \
    inline auto& CZ = ::scaluq::gate::CZ<Prec, Space>;                                             \
    inline auto& CCX = ::scaluq::gate::CCX<Prec, Space>;                                           \
    inline auto& Toffoli = ::scaluq::gate::Toffoli<Prec, Space>;                                   \
    inline auto& CCNot = ::scaluq::gate::CCNot<Prec, Space>;                                       \
    inline Gate Swap(std::uint64_t target1,                                                        \
                     std::uint64_t target2,                                                        \
                     const std::vector<std::uint64_t>& controls = {},                              \
                     std::vector<std::uint64_t> control_values = {}) {                             \
        return ::scaluq::gate::Swap<Prec, Space>(target1, target2, controls, control_values);      \
    }                                                                                              \
    inline Gate Pauli(const PauliOperator& pauli,                                                  \
                      const std::vector<std::uint64_t>& controls = {},                             \
                      std::vector<std::uint64_t> control_values = {}) {                            \
        return ::scaluq::gate::Pauli<Prec, Space>(pauli, controls, control_values);                \
    }                                                                                              \
    inline Gate PauliRotation(const PauliOperator& pauli,                                          \
                              double angle,                                                        \
                              const std::vector<std::uint64_t>& controls = {},                     \
                              std::vector<std::uint64_t> control_values = {}) {                    \
        return ::scaluq::gate::PauliRotation<Prec, Space>(pauli, angle, controls, control_values); \
    }                                                                                              \
    inline Gate DenseMatrix(const std::vector<std::uint64_t>& targets,                             \
                            const ::scaluq::ComplexMatrix& matrix,                                 \
                            const std::vector<std::uint64_t>& controls = {},                       \
                            std::vector<std::uint64_t> control_values = {},                        \
                            bool is_unitary = false) {                                             \
        return ::scaluq::gate::DenseMatrix<Prec, Space>(                                           \
            targets, matrix, controls, control_values, is_unitary);                                \
    }                                                                                              \
    inline Gate SparseMatrix(const std::vector<std::uint64_t>& targets,                            \
                             const ::scaluq::SparseComplexMatrix& matrix,                          \
                             const std::vector<std::uint64_t>& controls = {},                      \
                             std::vector<std::uint64_t> control_values = {}) {                     \
        return ::scaluq::gate::SparseMatrix<Prec, Space>(                                          \
            targets, matrix, controls, control_values);                                            \
    }                                                                                              \
    inline auto& Probabilistic = ::scaluq::gate::Probabilistic<Prec, Space>;                       \
    inline auto& BitFlipNoise = ::scaluq::gate::BitFlipNoise<Prec, Space>;                         \
    inline auto& DephasingNoise = ::scaluq::gate::DephasingNoise<Prec, Space>;                     \
    inline auto& BitFlipAndDephasingNoise = ::scaluq::gate::BitFlipAndDephasingNoise<Prec, Space>; \
    inline auto& DepolarizingNoise = ::scaluq::gate::DepolarizingNoise<Prec, Space>;               \
    }                                                                                              \
    using ParamGate = ::scaluq::ParamGate<Prec, Space>;                                            \
    using ParamRXGate = ::scaluq::ParamRXGate<Prec, Space>;                                        \
    using ParamRYGate = ::scaluq::ParamRYGate<Prec, Space>;                                        \
    using ParamRZGate = ::scaluq::ParamRZGate<Prec, Space>;                                        \
    using ParamPauliRotationGate = ::scaluq::ParamPauliRotationGate<Prec, Space>;                  \
    using ParamProbabilisticGate = ::scaluq::ParamProbabilisticGate<Prec, Space>;                  \
    namespace gate {                                                                               \
    inline ParamGate ParamRX(std::uint64_t target,                                                 \
                             double param_coef = 1.,                                               \
                             const std::vector<std::uint64_t>& controls = {},                      \
                             std::vector<std::uint64_t> control_values = {}) {                     \
        return ::scaluq::gate::ParamRX<Prec, Space>(target, param_coef, controls, control_values); \
    }                                                                                              \
    inline ParamGate ParamRY(std::uint64_t target,                                                 \
                             double param_coef = 1.,                                               \
                             const std::vector<std::uint64_t>& controls = {},                      \
                             std::vector<std::uint64_t> control_values = {}) {                     \
        return ::scaluq::gate::ParamRY<Prec, Space>(target, param_coef, controls, control_values); \
    }                                                                                              \
    inline ParamGate ParamRZ(std::uint64_t target,                                                 \
                             double param_coef = 1.,                                               \
                             const std::vector<std::uint64_t>& controls = {},                      \
                             std::vector<std::uint64_t> control_values = {}) {                     \
        return ::scaluq::gate::ParamRZ<Prec, Space>(target, param_coef, controls, control_values); \
    }                                                                                              \
    inline ParamGate ParamPauliRotation(const PauliOperator& pauli,                                \
                                        double param_coef = 1.,                                    \
                                        const std::vector<std::uint64_t>& controls = {},           \
                                        std::vector<std::uint64_t> control_values = {}) {          \
        return ::scaluq::gate::ParamPauliRotation<Prec, Space>(                                    \
            pauli, param_coef, controls, control_values);                                          \
    }                                                                                              \
    const auto& ParamProbabilistic = ::scaluq::gate::ParamProbabilistic<Prec, Space>;              \
    }                                                                                              \
    inline auto& merge_gate = ::scaluq::merge_gate<Prec, Space>;                                   \
    using Circuit = ::scaluq::Circuit<Prec, Space>;
