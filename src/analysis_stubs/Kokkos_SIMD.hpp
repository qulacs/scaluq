#pragma once
// Minimal Kokkos SIMD stub for clang-tidy static analysis.

#include <cstddef>

namespace Kokkos::Experimental {

struct vector_aligned_tag {};

template <class T>
class simd {
public:
    using value_type = T;

    simd() = default;
    explicit simd(T value) : _value(value) {}

    template <class Generator>
    explicit simd(Generator generator) : _value(generator(0)) {}

    simd(const T* ptr, vector_aligned_tag) : _value(*ptr) {}

    static constexpr std::size_t size() { return 1; }

    T operator[](std::size_t) const { return _value; }

    friend simd operator+(const simd& lhs, const simd& rhs) {
        return simd(lhs._value + rhs._value);
    }
    friend simd operator*(const simd& lhs, const simd& rhs) {
        return simd(lhs._value * rhs._value);
    }

    void copy_to(T* ptr, vector_aligned_tag) const { *ptr = _value; }

private:
    T _value{};
};

}  // namespace Kokkos::Experimental
