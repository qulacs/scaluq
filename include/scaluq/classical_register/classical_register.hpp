#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef SCALUQ_USE_NANOBIND
#include "../types.hpp"
#endif

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

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
inline void bind_classical_register_hpp(nb::module_& m) {
    using namespace nb::literals;

    nb::class_<ClassicalRegister>(m, "ClassicalRegister", "Classical register.")
        .def(nb::init<std::uint64_t>(), "register_size"_a, "Initialize classical register.")
        .def("register_size", &ClassicalRegister::register_size, "Get register size.")
        .def("__len__", &ClassicalRegister::register_size, "Get register size.")
        .def(
            "__getitem__",
            [](const ClassicalRegister& classical_register, std::uint64_t index) {
                return classical_register[index];
            },
            "index"_a,
            "Get classical bit.")
        .def(
            "__setitem__",
            [](ClassicalRegister& classical_register, std::uint64_t index, bool value) {
                classical_register[index] = value;
            },
            "index"_a,
            "value"_a,
            "Set classical bit.")
        .def("reset", &ClassicalRegister::reset, "Reset all bits to `False`.");
}
}  // namespace internal
#endif

}  // namespace scaluq
