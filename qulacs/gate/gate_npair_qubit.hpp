#pragma once

#include <cassert>
#include <ranges>

#include "gate.hpp"

namespace qulacs {
namespace internal {
class FusedSWAPGateImpl : public GateBase {
    UINT _qubit_index1, _qubit_index2, _block_size;

public:
    FusedSWAPGateImpl(UINT qubit_index1, UINT qubit_index2, UINT block_size)
        : _qubit_index1(qubit_index1), _qubit_index2(qubit_index2), _block_size(block_size) {
        UINT upper_index = std::max(qubit_index1, qubit_index2);
        UINT lower_index = std::min(qubit_index1, qubit_index2);
        if (upper_index <= (lower_index + block_size - 1)) {
            throw std::runtime_error(
                "FusedSwap: upper index must be bigger than lower_index + block_size - 1");
        }
    };

    UINT qubit_index1() const { return _qubit_index1; }
    UINT qubit_index2() const { return _qubit_index2; }
    UINT block_size() const { return _block_size; }

    std::vector<UINT> get_target_qubit_list() const override {
        std::vector<UINT> res(_block_size * 2);
        std::ranges::copy(std::views::iota(_qubit_index1, _qubit_index1 + _block_size),
                          res.begin());
        std::ranges::copy(std::views::iota(_qubit_index2, _qubit_index2 + _block_size),
                          res.begin() + _block_size);
        return res;
    }
    std::vector<UINT> get_control_qubit_list() const override { return {}; }

    void update_quantum_state(StateVector& state_vector) const override;
};
}  // namespace internal

using FusedSWAPGate = internal::GatePtr<internal::FusedSWAPGateImpl>;
}  // namespace qulacs
