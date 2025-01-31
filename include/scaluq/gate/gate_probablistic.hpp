#pragma once

#include "../util/random.hpp"
#include "gate_matrix.hpp"
#include "gate_pauli.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {

<<<<<<< HEAD
template <Precision Prec>
class ProbablisticGateImpl : public GateBase<Prec> {
    std::vector<double> _distribution;
    std::vector<double> _cumulative_distribution;
    std::vector<Gate<Prec>> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<double>& distribution,
                         const std::vector<Gate<Prec>>& gate_list);
    const std::vector<Gate<Prec>>& gate_list() const { return _gate_list; }
    const std::vector<double>& distribution() const { return _distribution; }
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ProbablisticGateImpl : public GateBase<Fp, Sp> {
    std::vector<Fp> _distribution;
    std::vector<Fp> _cumulative_distribution;
    std::vector<Gate<Fp, Sp>> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<Fp>& distribution,
                         const std::vector<Gate<Fp, Sp>>& gate_list);
    const std::vector<Gate<Fp, Sp>>& gate_list() const { return _gate_list; }
    const std::vector<Fp>& distribution() const { return _distribution; }
>>>>>>> set-space

    std::vector<std::uint64_t> target_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::target_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> control_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::vector<std::uint64_t> operand_qubit_list() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::operand_qubit_list(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t target_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::target_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t control_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::control_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }
    std::uint64_t operand_qubit_mask() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::operand_qubit_mask(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override;
    ComplexMatrix get_matrix() const override {
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override;
    internal::ComplexMatrix<Fp> get_matrix() const override {
>>>>>>> set-space
        throw std::runtime_error(
            "ProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Probablistic"},
                 {"gate_list", this->gate_list()},
                 {"distribution", this->distribution()}};
    }
};
}  // namespace internal

<<<<<<< HEAD
template <Precision Prec>
using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl<Prec>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION(Prec)                                 \
=======
template <std::floating_point Fp, ExecutionSpace Sp>
using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl<Fp, Sp>>;

namespace internal {

/*#define DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(Type)                                      \
>>>>>>> set-space
    template <>                                                                             \
    inline std::shared_ptr<const ProbablisticGateImpl<Prec>> get_from_json(const Json& j) { \
        auto distribution = j.at("distribution").get<std::vector<double>>();                \
        auto gate_list = j.at("gate_list").get<std::vector<Gate<Prec>>>();                  \
        return std::make_shared<const ProbablisticGateImpl<Prec>>(distribution, gate_list); \
    }

<<<<<<< HEAD
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_PROBGATE_WITH_PRECISION
=======
DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE*/
>>>>>>> set-space

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_probablistic(nb::module_& m) {
    DEF_GATE(ProbablisticGate,
             Prec,
             "Specific class of probablistic gate. The gate to apply is picked from a cirtain "
             "distribution.")
        .def(
            "gate_list",
<<<<<<< HEAD
            [](const ProbablisticGate<Prec>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate<Prec>& gate) { return gate->distribution(); },
=======
            [](const ProbablisticGate<Fp, Sp>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate<Fp, Sp>& gate) { return gate->distribution(); },
>>>>>>> set-space
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
