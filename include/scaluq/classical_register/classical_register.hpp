#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace scaluq {

class ClassicalRegister {
private:
    std::vector<bool> _bits;

public:
    explicit ClassicalRegister(std::uint64_t size) : _bits(size, false) {}

    [[nodiscard]] std::vector<bool>::reference operator[](std::uint64_t index) {
        if (index >= _bits.size()) {
            throw std::runtime_error(
                "ClassicalRegister::operator[](std::uint64_t): index out of bounds");
        }
        return _bits[index];
    }
    [[nodiscard]] bool operator[](std::uint64_t index) const {
        if (index >= _bits.size()) {
            throw std::runtime_error(
                "ClassicalRegister::operator[](std::uint64_t): index out of bounds");
        }
        return _bits[index];
    }
    [[nodiscard]] std::uint64_t register_size() const { return _bits.size(); }

    void reset() {
        for (auto&& bit : _bits) bit = false;
    }
};

}  // namespace scaluq
