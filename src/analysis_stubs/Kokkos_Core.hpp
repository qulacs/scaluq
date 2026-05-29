#pragma once
#ifndef KOKKOS_CORE_HPP
#define KOKKOS_CORE_HPP
#endif
// Minimal Kokkos stub for clang-tidy static analysis.

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#define KOKKOS_INLINE_FUNCTION       inline
#define KOKKOS_FORCEINLINE_FUNCTION  inline
#define KOKKOS_FUNCTION              inline
#define KOKKOS_LAMBDA                [=]
#define KOKKOS_CLASS_LAMBDA          [=, *this]

namespace Kokkos {

// ── Memory / execution space stubs ─────────────────────────────────────────
// Forward-declare HostSpace so Serial/OpenMP can reference it in memory_space.
struct HostSpace;
struct SharedSpace;

struct Serial {
    using memory_space    = HostSpace;
    using execution_space = Serial;
    Serial() = default;
};
struct OpenMP {
    using memory_space    = HostSpace;
    using execution_space = OpenMP;
    OpenMP() = default;
};
struct HostSpace   {};
struct SharedSpace {};

using DefaultHostExecutionSpace = OpenMP;
using DefaultExecutionSpace     = OpenMP;

// ── Layout tags ────────────────────────────────────────────────────────────
struct LayoutRight {};
struct LayoutLeft {};

// ── Memory traits ──────────────────────────────────────────────────────────
constexpr unsigned Unmanaged = 0x1u;
template<unsigned Flags = 0>
struct MemoryTraits {};
using MemoryUnmanaged = MemoryTraits<Unmanaged>;

// ── Allocation tag ─────────────────────────────────────────────────────────
struct ViewAllocateWithoutInitializing {
    const char* label;
    explicit ViewAllocateWithoutInitializing(const char* l = "") : label(l) {}
    explicit ViewAllocateWithoutInitializing(const std::string& s) : label(s.c_str()) {}
};

// ── Rank tag ───────────────────────────────────────────────────────────────
template<int N, class Iter = void, class Work = void>
struct Rank {};

// ── Sentinel tags ──────────────────────────────────────────────────────────
struct ALL_t {};
inline ALL_t ALL() { return {}; }

struct AUTO_t {};
inline constexpr AUTO_t AUTO{};

// ── pair ───────────────────────────────────────────────────────────────────
template<class First, class Second = First>
struct pair { First first; Second second; };

template<class T1, class T2>
inline pair<T1, T2> make_pair(T1 a, T2 b) { return {a, b}; }

// ── complex ────────────────────────────────────────────────────────────────
template<class T>
struct complex {
    T _r{}, _i{};
    complex() = default;
    complex(T r, T i = T{}) : _r(r), _i(i) {}
    complex(std::complex<T> c) : _r(c.real()), _i(c.imag()) {}
    template<class U> complex(const complex<U>& o) : _r(T(o._r)), _i(T(o._i)) {}
    T real() const { return _r; }
    T imag() const { return _i; }
    void real(T v) { _r = v; }
    void imag(T v) { _i = v; }
    operator std::complex<T>() const { return {_r, _i}; }
    complex& operator+=(const complex& o) { _r += o._r; _i += o._i; return *this; }
    complex& operator-=(const complex& o) { _r -= o._r; _i -= o._i; return *this; }
    complex& operator*=(const complex& o) {
        T r = _r*o._r - _i*o._i; _i = _r*o._i + _i*o._r; _r = r; return *this;
    }
};
template<class T> inline complex<T> operator+(complex<T> a, const complex<T>& b) { return a += b; }
template<class T> inline complex<T> operator*(complex<T> a, const complex<T>& b) { return a *= b; }
template<class T> inline complex<T> operator-(complex<T> a, const complex<T>& b) { return a -= b; }

// ── Array ──────────────────────────────────────────────────────────────────
template<class T, std::size_t N>
struct Array {
    T _elems[N]{};
    T&       operator[](std::size_t i)       { return _elems[i]; }
    const T& operator[](std::size_t i) const { return _elems[i]; }
    T*       data()       { return _elems; }
    const T* data() const { return _elems; }
    static constexpr std::size_t size() { return N; }
};

// ── View ───────────────────────────────────────────────────────────────────
namespace Impl {
    template<class T>           struct StripPointer             { using type = T; };
    template<class T>           struct StripPointer<T*>         { using type = T; };
    template<class T>           struct StripPointer<T**>        { using type = T; };
    template<class T>           struct StripPointer<const T*>   { using type = T; };
    template<class T>           struct StripPointer<const T**>  { using type = T; };
}

template<class DataType, class... Properties>
class View {
public:
    using value_type = typename Impl::StripPointer<DataType>::type;
    using pointer    = value_type*;
    using size_type  = std::size_t;
    using HostMirror = View<DataType, HostSpace>;

    View() = default;
    explicit View(const char*, size_type = 0, size_type = 0) {}
    explicit View(ViewAllocateWithoutInitializing, size_type = 0, size_type = 0) {}
    explicit View(pointer, size_type = 0, size_type = 0) {}
    explicit View(const value_type*, size_type = 0, size_type = 0) {}
    // Cross-space converting constructor: allows View<T*, HostSpace> to convert to View<T*, OpenMP>
    template<class... OtherProps>
    View(const View<DataType, OtherProps...>&) {}

    pointer   data()      const { return nullptr; }
    size_type extent(int) const { return 0; }
    size_type size()      const { return 0; }
    bool      is_null()   const { return true; }

    value_type& operator()()                      const { static value_type v{}; return v; }
    value_type& operator()(size_type)             const { static value_type v{}; return v; }
    value_type& operator()(size_type, size_type)  const { static value_type v{}; return v; }
    value_type& operator[](size_type)             const { static value_type v{}; return v; }
};

// ── Policies ───────────────────────────────────────────────────────────────
template<class ExecSpace = DefaultExecutionSpace>
struct RangePolicy {
    RangePolicy(std::size_t, std::size_t) {}
};

template<class... Properties>
struct MDRangePolicy {
    template<class A, class B>
    MDRangePolicy(A, B) {}
    template<class T, class U>
    MDRangePolicy(std::initializer_list<T>, std::initializer_list<U>) {}
};

template<class ExecSpace = DefaultExecutionSpace>
struct TeamPolicy {
    struct member_type {
        int  league_rank()  const { return 0; }
        int  team_rank()    const { return 0; }
        int  team_size()    const { return 1; }
        void team_barrier() const {}
    };
    TeamPolicy(ExecSpace, std::size_t, AUTO_t) {}
    TeamPolicy(std::size_t, AUTO_t) {}
};

// ── Thread-range helpers ───────────────────────────────────────────────────
struct TeamThreadRange {
    template<class M> TeamThreadRange(M&&, std::size_t) {}
    template<class M> TeamThreadRange(M&&, std::size_t, std::size_t) {}
};
struct ThreadVectorRange {
    template<class M> ThreadVectorRange(M&&, std::size_t) {}
    template<class M> ThreadVectorRange(M&&, std::size_t, std::size_t) {}
};
template<class... Props>
struct TeamThreadMDRange {
    template<class M, class... Args> TeamThreadMDRange(M&&, Args...) {}
};
struct PerTeam {
    template<class M> explicit PerTeam(M&&) {}
};

// ── Parallel dispatch ──────────────────────────────────────────────────────
template<class Policy, class Functor>
inline void parallel_for(Policy, Functor) {}
template<class Label, class Policy, class Functor>
inline void parallel_for(Label, Policy, Functor) {}
template<class Policy, class Functor, class Result>
inline void parallel_reduce(Policy, Functor, Result&) {}
template<class Policy, class Functor, class Result>
inline void parallel_reduce(Policy, Functor, Result&&) {}
template<class Label, class Policy, class Functor, class Result>
inline void parallel_reduce(Label, Policy, Functor, Result&) {}
template<class Label, class Policy, class Functor, class Result>
inline void parallel_reduce(Label, Policy, Functor, Result&&) {}
template<class Policy, class Functor>
inline void parallel_scan(Policy, Functor) {}
template<class Label, class Policy, class Functor>
inline void parallel_scan(Label, Policy, Functor) {}

template<class Scope, class Closure>
inline void single(Scope, Closure&&) {}

// ── Memory / view utilities ────────────────────────────────────────────────
template<class D, class S>
inline void deep_copy(D&&, S&&) {}

template<class V>
inline auto create_mirror_view(const V& v) { return v; }

template<class Space, class V>
inline auto create_mirror_view(Space, const V& v) { return v; }

template<class Space, class V>
inline auto create_mirror_view_and_copy(Space, const V& v) { return v; }

// subview strips one dimension: returns a 1D View of the element type.
template<class V, class... Idx>
inline auto subview(const V&, Idx...) {
    return View<typename V::value_type*>{};
}

// ── Lifecycle ─────────────────────────────────────────────────────────────
inline void fence() {}
inline void initialize() {}
inline void initialize(int, char**) {}
inline void finalize() {}
inline bool is_initialized() { return true; }
inline bool is_finalized()   { return false; }

// ── Atomic / bit ops ──────────────────────────────────────────────────────
template<class T>
inline void atomic_add(T* p, T v) { *p += v; }

template<class T>
inline void kokkos_swap(T& a, T& b) { T t = a; a = b; b = t; }

template<class T>
inline T max(T a, T b) { return a > b ? a : b; }

inline std::uint64_t popcount(std::uint64_t x) {
    return static_cast<std::uint64_t>(__builtin_popcountll(x));
}
inline std::uint64_t bit_width(std::uint64_t x) {
    return x == 0 ? 0 : static_cast<std::uint64_t>(64 - __builtin_clzll(x));
}

// ── Math constants ─────────────────────────────────────────────────────────
namespace numbers {
    inline constexpr double pi    = 3.14159265358979323846;
    inline constexpr double sqrt2 = 1.41421356237309504880;
    template<class T> inline constexpr T pi_v    = static_cast<T>(pi);
    template<class T> inline constexpr T sqrt2_v = static_cast<T>(sqrt2);
}

} // namespace Kokkos
