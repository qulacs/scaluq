#pragma once

#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec>
class MeasurementGateImpl : public GateBase<Prec> {
    std::uint64_t _classical_bit_index;
    bool _reset;

public:
    using GateBase<Prec>::update_quantum_state;

    MeasurementGateImpl(std::uint64_t target_mask, std::uint64_t classical_bit_index, bool reset)
        : GateBase<Prec>(target_mask, 0, 0),
          _classical_bit_index(classical_bit_index),
          _reset(reset) {}

    [[nodiscard]] std::uint64_t classical_bit_index() const { return _classical_bit_index; }
    [[nodiscard]] bool reset() const { return _reset; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        throw std::runtime_error(
            "Measurement::get_inverse: Measurement gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(ExecutionContext<Prec, ExecutionSpace::Host> context) const override;
    void update_quantum_state(
        ExecutionContextBatched<Prec, ExecutionSpace::Host> context) const override;
    void update_quantum_state(
        ExecutionContext<Prec, ExecutionSpace::HostSerial> context) const override;
    void update_quantum_state(
        ExecutionContextBatched<Prec, ExecutionSpace::HostSerial> context) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        ExecutionContext<Prec, ExecutionSpace::Default> context) const override;
    void update_quantum_state(
        ExecutionContextBatched<Prec, ExecutionSpace::Default> context) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Measurement"},
                 {"target", this->target_qubit_list()},
                 {"classical_bit", _classical_bit_index},
                 {"reset", _reset}};
    }
};

}  // namespace internal

template <Precision Prec>
using MeasurementGate = internal::GatePtr<internal::MeasurementGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_measurement_hpp(nb::module_& m, nb::class_<Gate<Prec>>& gate_base_def) {
    bind_specific_gate<MeasurementGate<Prec>, Prec>(
        m,
        gate_base_def,
        "MeasurementGate",
        "Specific class of computational-basis measurement gate.\n\nNotes:\n\tThis gate is "
        "not unitary and requires a classical register when applied.")
        .def(
            "classical_bit_index",
            [](const MeasurementGate<Prec>& gate) { return gate->classical_bit_index(); },
            "Get `classical_bit_index` property.")
        .def(
            "reset",
            [](const MeasurementGate<Prec>& gate) { return gate->reset(); },
            "Return whether this measurement resets the target qubit to |0>.");
}
}  // namespace internal
#endif
}  // namespace scaluq
