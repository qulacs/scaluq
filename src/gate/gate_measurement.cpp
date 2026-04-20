#include <bit>
#include <scaluq/gate/gate_measurement.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
ComplexMatrix MeasurementGateImpl<Prec>::get_matrix() const {
    throw std::runtime_error(
        "MeasurementGate::get_matrix(): measurement gate does not have a matrix representation");
}
template <Precision Prec>
std::string MeasurementGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Measurement\n";
    ss << indent << "  Classical Bit Index: " << _classical_bit_index << "\n";
    ss << indent << "  Reset: " << (_reset ? "true" : "false") << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}

namespace {
template <Precision Prec, ExecutionSpace Space>
bool apply_measurement_update(StateVector<Prec, Space>& state,
                              std::mt19937_64& random_engine,
                              std::uint64_t target_mask,
                              bool reset) {
    const std::uint64_t target = std::countr_zero(target_mask);
    if (reset) {
        const double zero_probability = state.get_zero_probability(target);
        std::vector<std::uint64_t> one_values(state.n_qubits(),
                                              StateVector<Prec, Space>::UNMEASURED);
        one_values[target] = 1;
        const double one_probability = state.get_marginal_probability(one_values);
        if (zero_probability != 0.0 && one_probability != 0.0) {
            throw std::runtime_error(
                "MeasurementGate::update_quantum_state(): reset target qubit is neither |0> nor "
                "|1>.");
        }
    }
    const double zero_probability = state.get_zero_probability(target);
    std::bernoulli_distribution zero_distribution(zero_probability);
    const bool measured_zero = zero_distribution(random_engine);
    if (measured_zero) {
        p0_gate(target_mask, 0, 0, state);
    } else {
        p1_gate(target_mask, 0, 0, state);
    }
    state.normalize();
    if (reset && !measured_zero) {
        x_gate(target_mask, 0, 0, state);
    }
    return measured_zero;
}

}  // namespace

#define DEFINE_MEASUREMENT_GATE_UPDATE(Space)                                                   \
    template <Precision Prec>                                                                   \
    void MeasurementGateImpl<Prec>::update_quantum_state(ExecutionContext<Prec, Space> context) \
        const {                                                                                 \
        this->check_qubit_mask_within_bounds(context.state);                                    \
        if (context.classical_register.register_size() <= _classical_bit_index) {               \
            throw std::runtime_error(                                                           \
                "MeasurementGate::update_quantum_state(): classical register size is too "      \
                "small for the requested classical bit index.");                                \
        }                                                                                       \
        const bool measured_zero = apply_measurement_update(                                    \
            context.state, context.random_engine, this->_target_mask, _reset);                  \
        context.classical_register[_classical_bit_index] = !measured_zero;                      \
    }                                                                                           \
    template <Precision Prec>                                                                   \
    void MeasurementGateImpl<Prec>::update_quantum_state(                                       \
        ExecutionContextBatched<Prec, Space> context) const {                                   \
        this->check_qubit_mask_within_bounds(context.states);                                   \
        if (context.classical_register.batch_size() != context.states.batch_size()) {           \
            throw std::runtime_error(                                                           \
                "MeasurementGate::update_quantum_state(): batch size mismatch.");               \
        }                                                                                       \
        if (context.classical_register.register_size() <= _classical_bit_index) {               \
            throw std::runtime_error(                                                           \
                "MeasurementGate::update_quantum_state(): classical register size is too "      \
                "small for the requested classical bit index.");                                \
        }                                                                                       \
        for (std::uint64_t batch_index = 0; batch_index < context.states.batch_size();          \
             ++batch_index) {                                                                   \
            auto state = context.states.view_state_vector_at(batch_index);                      \
            const bool measured_zero = apply_measurement_update(                                \
                state, context.random_engine, this->_target_mask, _reset);                      \
            context.classical_register[batch_index][_classical_bit_index] = !measured_zero;     \
        }                                                                                       \
    }
DEFINE_MEASUREMENT_GATE_UPDATE(ExecutionSpace::Host)
DEFINE_MEASUREMENT_GATE_UPDATE(ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_MEASUREMENT_GATE_UPDATE(ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_MEASUREMENT_GATE_UPDATE
template class MeasurementGateImpl<Prec>;

template <Precision Prec>
std::shared_ptr<const MeasurementGateImpl<Prec>> GetGateFromJson<MeasurementGateImpl<Prec>>::get(
    const Json& j) {
    const bool reset = j.contains("reset") ? j.at("reset").get<bool>() : false;
    return std::make_shared<const MeasurementGateImpl<Prec>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        j.at("classical_bit").get<std::uint64_t>(),
        reset);
}
template struct GetGateFromJson<MeasurementGateImpl<Prec>>;

}  // namespace scaluq::internal
