#pragma once
// Minimal Kokkos SIMD stub for clang-tidy static analysis.

#include <cstddef>

namespace Kokkos::Experimental {

struct vector_aligned_tag {};
inline constexpr vector_aligned_tag simd_flag_aligned{};

template <class T, std::size_t = 0>
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

template <class Simd>
Simd simd_unchecked_load(const typename Simd::value_type* ptr, vector_aligned_tag) {
    return Simd(ptr, vector_aligned_tag{});
}

template <class T, std::size_t N>
void simd_unchecked_store(const simd<T, N>& value, T* ptr, vector_aligned_tag) {
    value.copy_to(ptr, vector_aligned_tag{});
}

template <class SimdType>
SimdType simd_unchecked_load(const typename SimdType::value_type* ptr,
                             vector_aligned_tag flag = {}) {
    return SimdType(ptr, flag);
}

template <class T, std::size_t N>
void simd_unchecked_store(const simd<T, N>& value, T* ptr, vector_aligned_tag flag = {}) {
    value.copy_to(ptr, flag);
}

}  // namespace Kokkos::Experimental

namespace Kokkos {

template <class T, std::size_t N>
Experimental::simd<T, N> fma(const Experimental::simd<T, N>& x,
                             const Experimental::simd<T, N>& y,
                             const Experimental::simd<T, N>& z) {
    return x * y + z;
}

}  // namespace Kokkos
