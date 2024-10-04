#include <Kokkos_Core.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "state/state_vector.hpp"

namespace scaluq {
namespace internal {

enum class GateType { Unknown, X };

class XGateImpl;

template <typename T>
constexpr GateType get_gate_type() {
    if constexpr (std::is_same_v<T, XGateImpl>) {
        return GateType::X;
    } else {
        static_assert(lazy_false_v<T>, "unknown GateImpl");
    }
}

class GateBase : public std::enable_shared_from_this<GateBase> {
protected:
    std::uint64_t _target_mask, _control_mask;

    template <std::floating_point FloatType>
    void check_qubit_mask_within_bounds(const StateVector<FloatType>& state_vector) const {
        std::uint64_t full_mask = (1ULL << state_vector.n_qubits()) - 1;
        if ((_target_mask | _control_mask) > full_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::update_quantum_state(StateVector& state): "
                "Target/Control qubit exceeds the number of qubits in the system.");
        }
    }

    std::string get_qubit_info_as_string(const std::string& indent) const {
        std::ostringstream ss;
        auto targets = target_qubit_list();
        auto controls = control_qubit_list();
        ss << indent << "  Target Qubits: {";
        for (std::uint32_t i = 0; i < targets.size(); ++i)
            ss << targets[i] << (i == targets.size() - 1 ? "" : ", ");
        ss << "}\n";
        ss << indent << "  Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        ss << "}";
        return ss.str();
    }

public:
    GateBase(std::uint64_t target_mask, std::uint64_t control_mask)
        : _target_mask(target_mask), _control_mask(control_mask) {
        if (_target_mask & _control_mask) [[unlikely]] {
            throw std::runtime_error(
                "Error: Gate::Gate(std::uint64_t target_mask, std::uint64_t control_mask) : Target "
                "and control qubits must not overlap.");
        }
    }
    virtual ~GateBase() = default;

    [[nodiscard]] virtual std::vector<std::uint64_t> target_qubit_list() const {
        return mask_to_vector(_target_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> control_qubit_list() const {
        return mask_to_vector(_control_mask);
    }
    [[nodiscard]] virtual std::vector<std::uint64_t> operand_qubit_list() const {
        return mask_to_vector(_target_mask | _control_mask);
    }
    [[nodiscard]] virtual std::uint64_t target_qubit_mask() const { return _target_mask; }
    [[nodiscard]] virtual std::uint64_t control_qubit_mask() const { return _control_mask; }
    [[nodiscard]] virtual std::uint64_t operand_qubit_mask() const {
        return _target_mask | _control_mask;
    }

    virtual void update_quantum_state(StateVector<double>& state_vector) const = 0;
    virtual void update_quantum_state(StateVector<float>& state_vector) const = 0;

    [[nodiscard]] virtual std::string to_string(const std::string& indent = "") const = 0;
};

template <typename T>
concept GateImpl = std::derived_from<T, GateBase>;

template <GateImpl T>
class GatePtr {
private:
    std::shared_ptr<const T> _gate_ptr;
    GateType _gate_type;

public:
    GatePtr() : _gate_ptr(nullptr), _gate_type(GateType::Unknown) {}
    GatePtr(const GatePtr& gate) = default;

    template <typename U>
    GatePtr(const std::shared_ptr<const U>& gate_ptr) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = get_gate_type<U>();
            _gate_ptr = gate_ptr;
        } else {
            if constexpr (std::is_base_of_v<T, U>) {
                _gate_type = get_gate_type<U>();
                _gate_ptr = std::static_pointer_cast<const T>(gate_ptr);
            } else {
                _gate_type = get_gate_type<T>();
                if (!(_gate_ptr = std::dynamic_pointer_cast<const T>(gate_ptr))) {
                    throw std::runtime_error("invalid gate cast");
                }
            }
        }
    }

    template <typename U>
    GatePtr(const GatePtr<U>& gate) {
        if constexpr (std::is_same_v<T, U>) {
            _gate_type = gate._gate_type;
            _gate_ptr = gate._gate_ptr;
        } else if constexpr (std::is_base_of_v<T, U>) {
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        } else {
            if (gate._gate_type != get_gate_type<T>()) {
                throw std::runtime_error("invalid gate cast");
            }
            _gate_type = gate._gate_type;
            _gate_ptr = std::static_pointer_cast<const T>(gate._gate_ptr);
        }
    }

    GateType gate_type() const { return _gate_type; }

    const T* operator->() const {
        if (!_gate_ptr) {
            throw std::runtime_error("GatePtr::operator->(): Gate is Null");
        }
        return _gate_ptr.get();
    }

    friend std::ostream& operator<<(std::ostream& os, const GatePtr& gate) {
        os << gate->to_string();
        return os;
    }
};

using Gate = GatePtr<GateBase>;

template <std::floating_point FloatType>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<FloatType>& state) {
    Kokkos::parallel_for(
        state.dim() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | target_mask]);
        });
    Kokkos::fence();
}

class XGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    void update_quantum_state(StateVector<double>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        x_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    void update_quantum_state(StateVector<float>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        x_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent = "") const override {
        std::ostringstream ss;
        ss << indent << "XGate";
        return ss.str();
    }
};

class GateFactory {
public:
    template <GateImpl T, typename... Args>
    static internal::Gate create_gate(Args... args) {
        return {std::make_shared<const T>(args...)};
    }
};

}  // namespace internal

namespace gate {

inline internal::Gate X(std::uint64_t target,
                        const std::vector<std::uint64_t>& control_qubits = {}) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(
        internal::vector_to_mask({target}), internal::vector_to_mask(control_qubits));
}

}  // namespace gate
}  // namespace scaluq

// 使用例
int main() {
    Kokkos::initialize();
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<double> state(n_qubits);
        state.load({0, 1, 2, 3, 4, 5, 6, 7});
        auto x_gate = scaluq::gate::X(1, {0, 2});
        x_gate->update_quantum_state(state);

        std::cout << state << std::endl;
    }
    Kokkos::finalize();
}
