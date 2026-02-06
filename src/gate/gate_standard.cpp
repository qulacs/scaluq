#include <scaluq/gate/gate_standard.hpp>

#include "update_ops.hpp"

namespace scaluq::internal {
template <Precision Prec>
ComplexMatrix IGateImpl<Prec>::get_matrix() const {
    return ComplexMatrix::Identity(1, 1);
}
template <Precision Prec>
std::string IGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: I\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_I_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void IGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        i_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_I_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_I_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_I_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_I_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_I_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_I_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_I_GATE_UPDATE
template class IGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix GlobalPhaseGateImpl<Prec>::get_matrix() const {
    return ComplexMatrix::Identity(1, 1) * std::exp(StdComplex(0, static_cast<double>(_phase)));
}
template <Precision Prec>
std::string GlobalPhaseGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: GlobalPhase\n";
    ss << indent << "  Phase: " << static_cast<double>(_phase) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_GLOBAL_PHASE_GATE_UPDATE(Class, Space)                                              \
    template <Precision Prec>                                                                      \
    void GlobalPhaseGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                                        \
        global_phase_gate(this->_target_mask,                                                      \
                          this->_control_mask,                                                     \
                          this->_control_value_mask,                                               \
                          this->_phase,                                                            \
                          state_vector);                                                           \
    }
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_GLOBAL_PHASE_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_GLOBAL_PHASE_GATE_UPDATE
template class GlobalPhaseGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix XGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, 1, 1, 0;
    return mat;
}
template <Precision Prec>
std::string XGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: X\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_X_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void XGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        x_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_X_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_X_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_X_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_X_GATE_UPDATE
template class XGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix YGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, StdComplex(0, -1), StdComplex(0, 1), 0;
    return mat;
}
template <Precision Prec>
std::string YGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Y\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_Y_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void YGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        y_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_Y_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_Y_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_Y_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_Y_GATE_UPDATE
template class YGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix ZGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, -1;
    return mat;
}
template <Precision Prec>
std::string ZGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Z\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_Z_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void ZGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        z_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_Z_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_Z_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_Z_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_Z_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_Z_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_Z_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_Z_GATE_UPDATE
template class ZGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix HGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 1, 1, -1;
    mat /= Kokkos::numbers::sqrt2;
    return mat;
}
template <Precision Prec>
std::string HGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: H\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_H_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void HGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        h_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_H_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_H_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_H_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_H_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_H_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_H_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_H_GATE_UPDATE
template class HGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(0, 1);
    return mat;
}
template <Precision Prec>
std::string SGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: S\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_S_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void SGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        s_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_S_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_S_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_S_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_S_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_S_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_S_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_S_GATE_UPDATE
template class SGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SdagGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(0, -1);
    return mat;
}
template <Precision Prec>
std::string SdagGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Sdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_S_DAG_GATE_UPDATE(Class, Space)                                                 \
    template <Precision Prec>                                                                  \
    void SdagGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {    \
        this->check_qubit_mask_within_bounds(state_vector);                                    \
        sdag_gate(                                                                             \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_S_DAG_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_S_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_S_DAG_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_S_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_S_DAG_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_S_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_S_DAG_GATE_UPDATE
template class SdagGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix TGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(1, 1) / Kokkos::numbers::sqrt2;
    return mat;
}
template <Precision Prec>
std::string TGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: T\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_T_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                     \
    void TGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                       \
        t_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_T_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_T_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_T_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_T_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_T_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_T_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_T_GATE_UPDATE
template class TGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix TdagGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, StdComplex(1, -1) / Kokkos::numbers::sqrt2;
    return mat;
}
template <Precision Prec>
std::string TdagGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Tdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_T_DAG_GATE_UPDATE(Class, Space)                                                 \
    template <Precision Prec>                                                                  \
    void TdagGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {    \
        this->check_qubit_mask_within_bounds(state_vector);                                    \
        tdag_gate(                                                                             \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_T_DAG_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_T_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_T_DAG_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_T_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_T_DAG_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_T_DAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_T_DAG_GATE_UPDATE
template class TdagGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SqrtXGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, .5), StdComplex(.5, -.5), StdComplex(.5, -.5), StdComplex(.5, .5);
    return mat;
}
template <Precision Prec>
std::string SqrtXGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtX\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_SQRT_X_GATE_UPDATE(Class, Space)                                                \
    template <Precision Prec>                                                                  \
    void SqrtXGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {   \
        this->check_qubit_mask_within_bounds(state_vector);                                    \
        sqrtx_gate(                                                                            \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_SQRT_X_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_SQRT_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_SQRT_X_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_SQRT_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SQRT_X_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_SQRT_X_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_SQRT_X_GATE_UPDATE
template class SqrtXGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SqrtXdagGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, -.5), StdComplex(.5, .5), StdComplex(.5, .5), StdComplex(.5, -.5);
    return mat;
}
template <Precision Prec>
std::string SqrtXdagGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtXdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_SQRT_XDAG_GATE_UPDATE(Class, Space)                                              \
    template <Precision Prec>                                                                   \
    void SqrtXdagGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                                     \
        sqrtxdag_gate(                                                                          \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);  \
    }
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_SQRT_XDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_SQRT_XDAG_GATE_UPDATE
template class SqrtXdagGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SqrtYGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, .5), StdComplex(-.5, -.5), StdComplex(.5, .5), StdComplex(.5, .5);
    return mat;
}
template <Precision Prec>
std::string SqrtYGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtY\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_SQRT_Y_GATE_UPDATE(Class, Space)                                                \
    template <Precision Prec>                                                                  \
    void SqrtYGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {   \
        this->check_qubit_mask_within_bounds(state_vector);                                    \
        sqrty_gate(                                                                            \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_SQRT_Y_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_SQRT_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_SQRT_Y_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_SQRT_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SQRT_Y_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_SQRT_Y_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_SQRT_Y_GATE_UPDATE
template class SqrtYGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SqrtYdagGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << StdComplex(.5, -.5), StdComplex(.5, -.5), StdComplex(-.5, .5), StdComplex(.5, -.5);
    return mat;
}
template <Precision Prec>
std::string SqrtYdagGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: SqrtYdag\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_SQRT_YDAG_GATE_UPDATE(Class, Space)                                              \
    template <Precision Prec>                                                                   \
    void SqrtYdagGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                                     \
        sqrtydag_gate(                                                                          \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector);  \
    }
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_SQRT_YDAG_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_SQRT_YDAG_GATE_UPDATE
template class SqrtYdagGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix P0GateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, 0;
    return mat;
}
template <Precision Prec>
std::string P0GateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P0\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_P0_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                      \
    void P0GateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                        \
        p0_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_P0_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_P0_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_P0_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_P0_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_P0_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_P0_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_P0_GATE_UPDATE
template class P0GateImpl<Prec>;

template <Precision Prec>
ComplexMatrix P1GateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 0, 0, 0, 1;
    return mat;
}
template <Precision Prec>
std::string P1GateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: P1\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_P1_GATE_UPDATE(Class, Space)                                                        \
    template <Precision Prec>                                                                      \
    void P1GateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {          \
        this->check_qubit_mask_within_bounds(state_vector);                                        \
        p1_gate(this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_P1_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_P1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_P1_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_P1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_P1_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_P1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_P1_GATE_UPDATE
template class P1GateImpl<Prec>;

template <Precision Prec>
ComplexMatrix RXGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::cos(half_angle), StdComplex(0, -std::sin(half_angle)),
        StdComplex(0, -std::sin(half_angle)), std::cos(half_angle);
    return mat;
}
template <Precision Prec>
std::string RXGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RX\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_RX_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void RXGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        rx_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_angle,                                                             \
                state_vector);                                                            \
    }
DEFINE_RX_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_RX_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_RX_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_RX_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_RX_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_RX_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_RX_GATE_UPDATE
template class RXGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix RYGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::cos(half_angle), -std::sin(half_angle), std::sin(half_angle), std::cos(half_angle);
    return mat;
}
template <Precision Prec>
std::string RYGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RY\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_RY_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void RYGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        ry_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_angle,                                                             \
                state_vector);                                                            \
    }
DEFINE_RY_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_RY_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_RY_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_RY_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_RY_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_RY_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_RY_GATE_UPDATE
template class RYGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix RZGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    double half_angle = static_cast<double>(this->_angle) / 2;
    mat << std::exp(StdComplex(0, -half_angle)), 0, 0, std::exp(StdComplex(0, half_angle));
    return mat;
}
template <Precision Prec>
std::string RZGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: RZ\n";
    ss << indent << "  Angle: " << static_cast<double>(this->_angle) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_RZ_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void RZGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        rz_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_angle,                                                             \
                state_vector);                                                            \
    }
DEFINE_RZ_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_RZ_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_RZ_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_RZ_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_RZ_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_RZ_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_RZ_GATE_UPDATE
template class RZGateImpl<Prec>;

template <Precision Prec>
ComplexMatrix U1GateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, std::exp(StdComplex(0, static_cast<double>(_lambda)));
    return mat;
}
template <Precision Prec>
std::string U1GateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U1\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_U1_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void U1GateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        u1_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_lambda,                                                            \
                state_vector);                                                            \
    }
DEFINE_U1_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_U1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_U1_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_U1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_U1_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_U1_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_U1_GATE_UPDATE
template class U1GateImpl<Prec>;

template <Precision Prec>
ComplexMatrix U2GateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << std::cos(Kokkos::numbers::pi / 4.),
        -std::exp(StdComplex(0, static_cast<double>(_lambda))) * std::sin(Kokkos::numbers::pi / 4),
        std::exp(StdComplex(0, static_cast<double>(_phi))) * std::sin(Kokkos::numbers::pi / 4),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::cos(Kokkos::numbers::pi / 4);
    return mat;
}
template <Precision Prec>
std::string U2GateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "  Phi: " << static_cast<double>(this->_phi) << "\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << indent << "Gate Type: U2\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_U2_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void U2GateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        u2_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_phi,                                                               \
                this->_lambda,                                                            \
                state_vector);                                                            \
    }
DEFINE_U2_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_U2_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_U2_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_U2_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_U2_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_U2_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_U2_GATE_UPDATE
template class U2GateImpl<Prec>;

template <Precision Prec>
ComplexMatrix U3GateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat(2, 2);
    mat << std::cos(static_cast<double>(_theta) / 2),
        -std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::sin(static_cast<double>(_theta) / 2),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::sin(static_cast<double>(_theta) / 2),
        std::exp(StdComplex(0, static_cast<double>(_phi))) *
            std::exp(StdComplex(0, static_cast<double>(_lambda))) *
            std::cos(static_cast<double>(_theta) / 2);
    return mat;
}
template <Precision Prec>
std::string U3GateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: U3\n";
    ss << indent << "  Theta: " << static_cast<double>(this->_theta) << "\n";
    ss << indent << "  Phi: " << static_cast<double>(this->_phi) << "\n";
    ss << indent << "  Lambda: " << static_cast<double>(this->_lambda) << "\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_U3_GATE_UPDATE(Class, Space)                                               \
    template <Precision Prec>                                                             \
    void U3GateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const { \
        this->check_qubit_mask_within_bounds(state_vector);                               \
        u3_gate(this->_target_mask,                                                       \
                this->_control_mask,                                                      \
                this->_control_value_mask,                                                \
                this->_theta,                                                             \
                this->_phi,                                                               \
                this->_lambda,                                                            \
                state_vector);                                                            \
    }
DEFINE_U3_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_U3_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_U3_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_U3_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_U3_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_U3_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_U3_GATE_UPDATE
template class U3GateImpl<Prec>;

template <Precision Prec>
ComplexMatrix SwapGateImpl<Prec>::get_matrix() const {
    ComplexMatrix mat = ComplexMatrix::Identity(1 << 2, 1 << 2);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    return mat;
}
template <Precision Prec>
std::string SwapGateImpl<Prec>::to_string(const std::string& indent) const {
    std::ostringstream ss;
    ss << indent << "Gate Type: Swap\n";
    ss << this->get_qubit_info_as_string(indent);
    return ss.str();
}
#define DEFINE_SWAP_GATE_UPDATE(Class, Space)                                                  \
    template <Precision Prec>                                                                  \
    void SwapGateImpl<Prec>::update_quantum_state(Class<Prec, Space>& state_vector) const {    \
        this->check_qubit_mask_within_bounds(state_vector);                                    \
        swap_gate(                                                                             \
            this->_target_mask, this->_control_mask, this->_control_value_mask, state_vector); \
    }
DEFINE_SWAP_GATE_UPDATE(StateVector, ExecutionSpace::Host)
DEFINE_SWAP_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Host)
DEFINE_SWAP_GATE_UPDATE(StateVector, ExecutionSpace::HostSerial)
DEFINE_SWAP_GATE_UPDATE(StateVectorBatched, ExecutionSpace::HostSerial)
#ifdef SCALUQ_USE_CUDA
DEFINE_SWAP_GATE_UPDATE(StateVector, ExecutionSpace::Default)
DEFINE_SWAP_GATE_UPDATE(StateVectorBatched, ExecutionSpace::Default)
#endif  // SCALUQ_USE_CUDA
#undef DEFINE_SWAP_GATE_UPDATE
template class SwapGateImpl<Prec>;

// I
template <Precision Prec>
std::shared_ptr<const IGateImpl<Prec>> GetGateFromJson<IGateImpl<Prec>>::get(const Json&) {
    return std::make_shared<const IGateImpl<Prec>>();
}
template struct GetGateFromJson<IGateImpl<Prec>>;

// GlobalPhase
template <Precision Prec>
std::shared_ptr<const GlobalPhaseGateImpl<Prec>> GetGateFromJson<GlobalPhaseGateImpl<Prec>>::get(
    const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const GlobalPhaseGateImpl<Prec>>(
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        static_cast<Float<Prec>>(j.at("phase").get<double>()));
}
template struct GetGateFromJson<GlobalPhaseGateImpl<Prec>>;

// X, Y, Z, H, S, Sdag, T, Tdag, SqrtX, SqrtY, P0, P1
#define DECLARE_GET_FROM_JSON_SINGLE_IMPL(Impl)                                         \
    template <Precision Prec>                                                           \
    std::shared_ptr<const Impl<Prec>> GetGateFromJson<Impl<Prec>>::get(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();        \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();  \
        return std::make_shared<const Impl<Prec>>(                                      \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),           \
            vector_to_mask(control_qubits),                                             \
            vector_to_mask(control_qubits, control_values));                            \
    }                                                                                   \
    template struct GetGateFromJson<Impl<Prec>>;
DECLARE_GET_FROM_JSON_SINGLE_IMPL(XGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(YGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(ZGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(HGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SdagGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(TGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(TdagGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtXGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtXdagGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtYGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtYdagGateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(P0GateImpl)
DECLARE_GET_FROM_JSON_SINGLE_IMPL(P1GateImpl)
#undef DECLARE_GET_FROM_JSON_SINGLE_IMPL

// RX, RY, RZ
#define DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(Impl)                                       \
    template <Precision Prec>                                                           \
    std::shared_ptr<const Impl<Prec>> GetGateFromJson<Impl<Prec>>::get(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();        \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();  \
        return std::make_shared<const Impl<Prec>>(                                      \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),           \
            vector_to_mask(control_qubits),                                             \
            vector_to_mask(control_qubits, control_values),                             \
            static_cast<Float<Prec>>(j.at("angle").get<double>()));                     \
    }                                                                                   \
    template struct GetGateFromJson<Impl<Prec>>;
DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RXGateImpl)
DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RYGateImpl)
DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RZGateImpl)
#undef DECLARE_GET_FROM_JSON_R_SINGLE_IMPL

// U1, U2, U3
template <Precision Prec>
std::shared_ptr<const U1GateImpl<Prec>> GetGateFromJson<U1GateImpl<Prec>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const U1GateImpl<Prec>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        static_cast<Float<Prec>>(j.at("theta").get<double>()));
}
template <Precision Prec>
std::shared_ptr<const U2GateImpl<Prec>> GetGateFromJson<U2GateImpl<Prec>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const U2GateImpl<Prec>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        static_cast<Float<Prec>>(j.at("theta").get<double>()),
        static_cast<Float<Prec>>(j.at("phi").get<double>()));
}
template <Precision Prec>
std::shared_ptr<const U3GateImpl<Prec>> GetGateFromJson<U3GateImpl<Prec>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const U3GateImpl<Prec>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values),
        static_cast<Float<Prec>>(j.at("theta").get<double>()),
        static_cast<Float<Prec>>(j.at("phi").get<double>()),
        static_cast<Float<Prec>>(j.at("lambda").get<double>()));
}
template struct GetGateFromJson<U1GateImpl<Prec>>;
template struct GetGateFromJson<U2GateImpl<Prec>>;
template struct GetGateFromJson<U3GateImpl<Prec>>;

// Swap
template <Precision Prec>
std::shared_ptr<const SwapGateImpl<Prec>> GetGateFromJson<SwapGateImpl<Prec>>::get(const Json& j) {
    auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();
    auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();
    return std::make_shared<const SwapGateImpl<Prec>>(
        vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),
        vector_to_mask(control_qubits),
        vector_to_mask(control_qubits, control_values));
}
template struct GetGateFromJson<SwapGateImpl<Prec>>;
}  // namespace scaluq::internal
