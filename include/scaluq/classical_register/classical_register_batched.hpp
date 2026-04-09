#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "classical_register.hpp"

namespace scaluq {

class ClassicalRegisterBatched {
private:
    std::vector<ClassicalRegister> _registers;
    std::uint64_t _register_size = 0;

public:
    ClassicalRegisterBatched(std::uint64_t register_size, std::uint64_t batch_size)
        : _registers(batch_size, ClassicalRegister(register_size)), _register_size(register_size) {}

    [[nodiscard]] std::uint64_t register_size() const { return _register_size; }
    [[nodiscard]] std::uint64_t batch_size() const { return _registers.size(); }

    [[nodiscard]] ClassicalRegister& operator[](std::uint64_t batch_index) {
        if (batch_index >= _registers.size()) {
            throw std::runtime_error("ClassicalRegisterBatched: batch index out of bounds");
        }
        return _registers[batch_index];
    }
    [[nodiscard]] const ClassicalRegister& operator[](std::uint64_t batch_index) const {
        if (batch_index >= _registers.size()) {
            throw std::runtime_error("ClassicalRegisterBatched: batch index out of bounds");
        }
        return _registers[batch_index];
    }

    [[nodiscard]] std::vector<bool>::reference operator()(std::uint64_t batch_index,
                                                          std::uint64_t bit_index) {
        return (*this)[batch_index][bit_index];
    }
    [[nodiscard]] bool operator()(std::uint64_t batch_index, std::uint64_t bit_index) const {
        return (*this)[batch_index][bit_index];
    }

    void reset() {
        for (auto&& reg : _registers) reg.reset();
    }
};

}  // namespace scaluq
