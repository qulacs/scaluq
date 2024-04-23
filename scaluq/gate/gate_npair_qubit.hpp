#pragma once

#include <cassert>
#include <ranges>

#include "gate.hpp"

namespace scaluq {
namespace internal {
class FusedSwapGateImpl : public GateBase {
    UINT _qubit_index1, _qubit_index2, _block_size;

public:
    FusedSwapGateImpl(UINT qubit_index1, UINT qubit_index2, UINT block_size)
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

    Gate copy() const override { return std::make_shared<FusedSwapGateImpl>(*this); }
    Gate get_inverse() const override { return std::make_shared<FusedSwapGateImpl>(*this); }
    std::optional<ComplexMatrix> get_matrix() const override {
        const UINT pow2_nq = 1ULL << _block_size;
        const UINT pow2_2nq = 1ULL << (_block_size * 2);
        auto mat = SparseComplexMatrix(pow2_2nq, pow2_2nq);
        mat.reserve(pow2_2nq);
        for (UINT i = 0; i < pow2_nq; i++) {
            for (UINT j = 0; j < pow2_nq; j++) {
                mat.insert(i * pow2_nq + j, i + j * pow2_nq) = 1;
            }
        }
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_qubit_index1 + this->_block_size - 1);
        check_qubit_within_bounds(state_vector, this->_qubit_index2 + this->_block_size - 1);
        fusedswap_gate(this->_qubit_index1, this->_qubit_index2, this->_block_size, state_vector);
    }
};
}  // namespace internal

using FusedSwapGate = internal::GatePtr<internal::FusedSwapGateImpl>;
}  // namespace scaluq
