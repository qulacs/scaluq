#pragma once

#include "../util/random.hpp"
#include "gate_matrix.hpp"
#include "gate_pauli.hpp"
#include "gate_standard.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
class ProbablisticGateImpl : public GateBase<Fp> {
    std::vector<Fp> _distribution;
    std::vector<Fp> _cumlative_distribution;
    std::vector<Gate<Fp>> _gate_list;

public:
    ProbablisticGateImpl(const std::vector<Fp>& distribution,
                         const std::vector<Gate<Fp>>& gate_list);
    const std::vector<Gate<Fp>>& gate_list() const { return _gate_list; }
    const std::vector<Fp>& distribution() const { return _distribution; }

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

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override;
    internal::ComplexMatrix<Fp> get_matrix() const override {
        throw std::runtime_error(
            "ProbablisticGateImpl::get_matrix(): This function must not be used in "
            "ProbablisticGateImpl.");
    }

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Probablistic"},
                 {"gate_list", this->gate_list()},
                 {"distribution", this->distribution()}};
    }
};
}  // namespace internal

template <std::floating_point Fp>
using ProbablisticGate = internal::GatePtr<internal::ProbablisticGateImpl<Fp>>;

namespace internal {

#define DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(Type)                                      \
    template <>                                                                             \
    inline std::shared_ptr<const ProbablisticGateImpl<Type>> get_from_json(const Json& j) { \
        auto distribution = j.at("distribution").get<std::vector<Type>>();                  \
        auto gate_list = j.at("gate_list").get<std::vector<Gate<Type>>>();                  \
        return std::make_shared<const ProbablisticGateImpl<Type>>(distribution, gate_list); \
    }

DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PROBGATE_WITH_TYPE(float)

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_gate_probablistic(nb::module_& m) {
    DEF_GATE(ProbablisticGate,
             Fp,
             "Specific class of probablistic gate. The gate to apply is picked from a cirtain "
             "distribution.")
        .def(
            "gate_list",
            [](const ProbablisticGate<Fp>& gate) { return gate->gate_list(); },
            nb::rv_policy::reference)
        .def(
            "distribution",
            [](const ProbablisticGate<Fp>& gate) { return gate->distribution(); },
            nb::rv_policy::reference);
}
}  // namespace internal
#endif
}  // namespace scaluq
