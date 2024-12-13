//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

/*
This file is a modified version of Kokkos, originally licensed under the Apache License v2.0 with
LLVM Exceptions. The modifications were made by Qulacs-Osaka on 2024/12/13. All the typename
`Kokkos::complex` was modified to `scaluq::Complex`. The other modifications are noted by comments.
*/

// Removed by Qulacs-Osaka on 2024/12/11
/*
#ifndef KOKKOS_COMPLEX_HPP
#define KOKKOS_COMPLEX_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#endif
*/

// Added by Qulacs-Osaka on 2024/12/13
#pragma once

#include <Kokkos_Atomic.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_NumericTraits.hpp>
#include <Kokkos_ReductionIdentity.hpp>
#include <complex>
#include <impl/Kokkos_Error.hpp>
#include <iosfwd>
#include <tuple>
#include <type_traits>

// Added by Qulacs-Osaka on 2024/12/11
#include "floating_point.hpp"

namespace scaluq {

/// \class Complex
/// \brief Partial reimplementation of std::complex that works as the
///   result of a Kokkos::parallel_reduce.
/// \tparam RealType The type of the real and imaginary parts of the
///   complex number.  As with std::complex, this is only defined for
///   \c float, \c double, and <tt>long double</tt>.  The latter is
///   currently forbidden in CUDA device kernels.
template <class RealType>
class
#ifdef KOKKOS_ENABLE_COMPLEX_ALIGN
    alignas(2 * sizeof(RealType))
#endif
        Complex {
    /*
        static_assert(std::is_floating_point_v<RealType> &&
                          std::is_same_v<RealType, std::remove_cv_t<RealType>>,
                      "Kokkos::complex can only be instantiated for a cv-unqualified "
                      "floating point type");
                      */
    static_assert(scaluq::FloatingPoint<RealType> &&
                      std::is_same_v<RealType, std::remove_cv_t<RealType>>,
                  "scaluq::Complex can only be instantiated for a cv-unqualified "
                  "floating point type");  // Modified by Qulacs-Osaka on 2024-12/11

private:
    RealType re_{};
    RealType im_{};

public:
    //! The type of the real or imaginary parts of this complex number.
    using value_type = RealType;

    //! Default constructor (initializes both real and imaginary parts to zero).
    KOKKOS_DEFAULTED_FUNCTION
    Complex() = default;

    //! Copy constructor.
    KOKKOS_DEFAULTED_FUNCTION
    Complex(const Complex&) noexcept = default;

    KOKKOS_DEFAULTED_FUNCTION
    Complex& operator=(const Complex&) noexcept = default;

    /// \brief Conversion constructor from compatible RType
    template <class RType, std::enable_if_t<std::is_convertible_v<RType, RealType>, int> = 0>
    KOKKOS_INLINE_FUNCTION Complex(const Complex<RType>& other) noexcept
        // Intentionally do the conversions implicitly here so that users don't
        // get any warnings about narrowing, etc., that they would expect to get
        // otherwise.
        : re_(other.real()), im_(other.imag()) {}

    /// \brief Conversion constructor from std::complex.
    ///
    /// This constructor cannot be called in a CUDA device function,
    /// because std::complex's methods and nonmember functions are not
    /// marked as CUDA device functions.
    KOKKOS_INLINE_FUNCTION
    Complex(const std::complex<RealType>& src) noexcept
        // We can use this aspect of the standard to avoid calling
        // non-device-marked functions `std::real` and `std::imag`: "For any
        // object z of type Complex<T>, reinterpret_cast<T(&)[2]>(z)[0] is the
        // real part of z and reinterpret_cast<T(&)[2]>(z)[1] is the imaginary
        // part of z." Now we don't have to provide a whole bunch of the overloads
        // of things taking either scaluq::Complex or std::complex
        : re_(reinterpret_cast<const RealType (&)[2]>(src)[0]),
          im_(reinterpret_cast<const RealType (&)[2]>(src)[1]) {}

    /// \brief Conversion operator to std::complex.
    ///
    /// This operator cannot be called in a CUDA device function,
    /// because std::complex's methods and nonmember functions are not
    /// marked as CUDA device functions.
    // TODO: make explicit.  DJS 2019-08-28
    operator std::complex<RealType>() const noexcept { return std::complex<RealType>(re_, im_); }

    /// \brief Constructor that takes just the real part, and sets the
    ///   imaginary part to zero.
    KOKKOS_INLINE_FUNCTION Complex(const RealType& val) noexcept
        : re_(val), im_(static_cast<RealType>(0)) {}

    //! Constructor that takes the real and imaginary parts.
    KOKKOS_INLINE_FUNCTION
    Complex(const RealType& re, const RealType& im) noexcept : re_(re), im_(im) {}

    //! Assignment operator (from a real number).
    KOKKOS_INLINE_FUNCTION Complex& operator=(const RealType& val) noexcept {
        re_ = val;
        im_ = RealType(0);
        return *this;
    }

    /// \brief Assignment operator from std::complex.
    ///
    /// This constructor cannot be called in a CUDA device function,
    /// because std::complex's methods and nonmember functions are not
    /// marked as CUDA device functions.
    Complex& operator=(const std::complex<RealType>& src) noexcept {
        *this = Complex(src);
        return *this;
    }

    //! The imaginary part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr RealType& imag() noexcept { return im_; }

    //! The real part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr RealType& real() noexcept { return re_; }

    //! The imaginary part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr RealType imag() const noexcept { return im_; }

    //! The real part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr RealType real() const noexcept { return re_; }

    //! Set the imaginary part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr void imag(RealType v) noexcept { im_ = v; }

    //! Set the real part of this complex number.
    KOKKOS_INLINE_FUNCTION
    constexpr void real(RealType v) noexcept { re_ = v; }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator+=(const Complex<RealType>& src) noexcept {
        re_ += src.re_;
        im_ += src.im_;
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator+=(const RealType& src) noexcept {
        re_ += src;
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator-=(const Complex<RealType>& src) noexcept {
        re_ -= src.re_;
        im_ -= src.im_;
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator-=(const RealType& src) noexcept {
        re_ -= src;
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator*=(const Complex<RealType>& src) noexcept {
        const RealType realPart = re_ * src.re_ - im_ * src.im_;
        const RealType imagPart = re_ * src.im_ + im_ * src.re_;
        re_ = realPart;
        im_ = imagPart;
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator*=(const RealType& src) noexcept {
        re_ *= src;
        im_ *= src;
        return *this;
    }

    // Conditional noexcept, just in case RType throws on divide-by-zero
    constexpr KOKKOS_INLINE_FUNCTION Complex& operator/=(const Complex<RealType>& y) noexcept(
        noexcept(RealType{} / RealType{})) {
        // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
        // If the real part is +/-Inf and the imaginary part is -/+Inf,
        // this won't change the result.
        const RealType s = fabs(y.real()) + fabs(y.imag());

        // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
        // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
        // because y/s is NaN.
        // TODO mark this branch unlikely
        if (s == RealType(0)) {
            this->re_ /= s;
            this->im_ /= s;
        } else {
            const Complex x_scaled(this->re_ / s, this->im_ / s);
            const Complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
            const RealType y_scaled_abs =
                y_conj_scaled.re_ * y_conj_scaled.re_ +
                y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
            *this = x_scaled * y_conj_scaled;
            *this /= y_scaled_abs;
        }
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator/=(const std::complex<RealType>& y) noexcept(
        noexcept(RealType{} / RealType{})) {
        // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
        // If the real part is +/-Inf and the imaginary part is -/+Inf,
        // this won't change the result.
        const RealType s = fabs(y.real()) + fabs(y.imag());

        // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
        // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
        // because y/s is NaN.
        if (s == RealType(0)) {
            this->re_ /= s;
            this->im_ /= s;
        } else {
            const Complex x_scaled(this->re_ / s, this->im_ / s);
            const Complex y_conj_scaled(y.re_ / s, -(y.im_) / s);
            const RealType y_scaled_abs =
                y_conj_scaled.re_ * y_conj_scaled.re_ +
                y_conj_scaled.im_ * y_conj_scaled.im_;  // abs(y) == abs(conj(y))
            *this = x_scaled * y_conj_scaled;
            *this /= y_scaled_abs;
        }
        return *this;
    }

    constexpr KOKKOS_INLINE_FUNCTION Complex& operator/=(const RealType& src) noexcept(
        noexcept(RealType{} / RealType{})) {
        re_ /= src;
        im_ /= src;
        return *this;
    }

    template <size_t I, typename RT>
    friend constexpr const RT& get(const Complex<RT>&) noexcept;

    template <size_t I, typename RT>
    friend constexpr const RT&& get(const Complex<RT>&&) noexcept;

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
    //! Copy constructor from volatile.
    template <class RType, std::enable_if_t<std::is_convertible_v<RType, RealType>, int> = 0>
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION Complex(const volatile Complex<RType>& src) noexcept
        // Intentionally do the conversions implicitly here so that users don't
        // get any warnings about narrowing, etc., that they would expect to get
        // otherwise.
        : re_(src.re_), im_(src.im_) {}

    /// \brief Assignment operator, for volatile <tt>*this</tt> and
    ///   nonvolatile input.
    ///
    /// \param src [in] Input; right-hand side of the assignment.
    ///
    /// This operator returns \c void instead of <tt>volatile
    /// Complex& </tt>.  See Kokkos Issue #177 for the
    /// explanation.  In practice, this means that you should not chain
    /// assignments with volatile lvalues.
    //
    // Templated, so as not to be a copy assignment operator (Kokkos issue #2577)
    // Intended to behave as
    //    void operator=(const Complex&) volatile noexcept
    //
    // Use cases:
    //    Complex r;
    //    const Complex cr;
    //    volatile Complex vl;
    //    vl = r;
    //    vl = cr;
    template <class Comp, std::enable_if_t<std::is_same_v<Comp, Complex>, int> = 0>
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator=(const Comp& src) volatile noexcept {
        re_ = src.re_;
        im_ = src.im_;
        // We deliberately do not return anything here.  See explanation
        // in public documentation above.
    }

    //! Assignment operator, volatile LHS and volatile RHS
    // TODO Should this return void like the other volatile assignment operators?
    //
    // Templated, so as not to be a copy assignment operator (Kokkos issue #2577)
    // Intended to behave as
    //    volatile Complex& operator=(const volatile Complex&) volatile noexcept
    //
    // Use cases:
    //    volatile Complex vr;
    //    const volatile Complex cvr;
    //    volatile Complex vl;
    //    vl = vr;
    //    vl = cvr;
    template <class Comp, std::enable_if_t<std::is_same_v<Comp, Complex>, int> = 0>
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION volatile Complex& operator=(
        const volatile Comp& src) volatile noexcept {
        re_ = src.re_;
        im_ = src.im_;
        return *this;
    }

    //! Assignment operator, volatile RHS and non-volatile LHS
    //
    // Templated, so as not to be a copy assignment operator (Kokkos issue #2577)
    // Intended to behave as
    //    Complex& operator=(const volatile Complex&) noexcept
    //
    // Use cases:
    //    volatile Complex vr;
    //    const volatile Complex cvr;
    //    Complex l;
    //    l = vr;
    //    l = cvr;
    //
    template <class Comp, std::enable_if_t<std::is_same_v<Comp, Complex>, int> = 0>
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION Complex& operator=(const volatile Comp& src) noexcept {
        re_ = src.re_;
        im_ = src.im_;
        return *this;
    }

    // Mirroring the behavior of the assignment operators from complex RHS in the
    // RealType RHS versions.

    //! Assignment operator (from a volatile real number).
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator=(const volatile RealType& val) noexcept {
        re_ = val;
        im_ = RealType(0);
        // We deliberately do not return anything here.  See explanation
        // in public documentation above.
    }

    //! Assignment operator volatile LHS and non-volatile RHS
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION Complex& operator=(
        const RealType& val) volatile noexcept {
        re_ = val;
        im_ = RealType(0);
        return *this;
    }

    //! Assignment operator volatile LHS and volatile RHS
    // TODO Should this return void like the other volatile assignment operators?
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION Complex& operator=(
        const volatile RealType& val) volatile noexcept {
        re_ = val;
        im_ = RealType(0);
        return *this;
    }

    //! The imaginary part of this complex number (volatile overload).
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION volatile RealType& imag() volatile noexcept {
        return im_;
    }

    //! The real part of this complex number (volatile overload).
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION volatile RealType& real() volatile noexcept {
        return re_;
    }

    //! The imaginary part of this complex number (volatile overload).
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION RealType imag() const volatile noexcept { return im_; }

    //! The real part of this complex number (volatile overload).
    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION RealType real() const volatile noexcept { return re_; }

    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator+=(
        const volatile Complex<RealType>& src) volatile noexcept {
        re_ += src.re_;
        im_ += src.im_;
    }

    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator+=(
        const volatile RealType& src) volatile noexcept {
        re_ += src;
    }

    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator*=(
        const volatile Complex<RealType>& src) volatile noexcept {
        const RealType realPart = re_ * src.re_ - im_ * src.im_;
        const RealType imagPart = re_ * src.im_ + im_ * src.re_;

        re_ = realPart;
        im_ = imagPart;
    }

    KOKKOS_DEPRECATED KOKKOS_INLINE_FUNCTION void operator*=(
        const volatile RealType& src) volatile noexcept {
        re_ *= src;
        im_ *= src;
    }
#endif  // KOKKOS_ENABLE_DEPRECATED_CODE_4
};

}  // namespace scaluq

// Tuple protocol for complex based on https://wg21.link/P2819R2 (voted into
// the C++26 working draft on 2023-11)

template <typename RealType>
struct std::tuple_size<scaluq::Complex<RealType>> : std::integral_constant<size_t, 2> {};

template <size_t I, typename RealType>
struct std::tuple_element<I, scaluq::Complex<RealType>> {
    static_assert(I < 2);
    using type = RealType;
};

namespace scaluq {

// get<...>(...) defined here so as not to be hidden friends, as per P2819R2

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr RealType& get(Complex<RealType>& z) noexcept {
    static_assert(I < 2);
    if constexpr (I == 0)
        return z.real();
    else
        return z.imag();
#ifdef KOKKOS_COMPILER_INTEL
    __builtin_unreachable();
#endif
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr RealType&& get(Complex<RealType>&& z) noexcept {
    static_assert(I < 2);
    if constexpr (I == 0)
        return std::move(z.real());
    else
        return std::move(z.imag());
#ifdef KOKKOS_COMPILER_INTEL
    __builtin_unreachable();
#endif
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr const RealType& get(const Complex<RealType>& z) noexcept {
    static_assert(I < 2);
    if constexpr (I == 0)
        return z.re_;
    else
        return z.im_;
#ifdef KOKKOS_COMPILER_INTEL
    __builtin_unreachable();
#endif
}

template <size_t I, typename RealType>
KOKKOS_FUNCTION constexpr const RealType&& get(const Complex<RealType>&& z) noexcept {
    static_assert(I < 2);
    if constexpr (I == 0)
        return std::move(z.re_);
    else
        return std::move(z.im_);
#ifdef KOKKOS_COMPILER_INTEL
    __builtin_unreachable();
#endif
}

//==============================================================================
// <editor-fold desc="Equality and inequality"> {{{1

// Note that this is not the same behavior as std::complex, which doesn't allow
// implicit conversions, but since this is the way we had it before, we have
// to do it this way now.

//! Binary == operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION bool operator==(Complex<RealType1> const& x,
                                       Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) == common_type(y.real()) &&
           common_type(x.imag()) == common_type(y.imag());
}

// TODO (here and elsewhere) decide if we should convert to a scaluq::Complex
//      and do the comparison in a device-marked function
//! Binary == operator for std::complex complex.
template <class RealType1, class RealType2>
inline bool operator==(std::complex<RealType1> const& x, Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) == common_type(y.real()) &&
           common_type(x.imag()) == common_type(y.imag());
}

//! Binary == operator for complex std::complex.
template <class RealType1, class RealType2>
inline bool operator==(Complex<RealType1> const& x, std::complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) == common_type(y.real()) &&
           common_type(x.imag()) == common_type(y.imag());
}

//! Binary == operator for complex real.
template <class RealType1,
          class RealType2,
          // Constraints to avoid participation in oparator==() for every possible RHS
          std::enable_if_t<std::is_convertible_v<RealType2, RealType1>, int> = 0>
KOKKOS_INLINE_FUNCTION bool operator==(Complex<RealType1> const& x, RealType2 const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) == common_type(y) && common_type(x.imag()) == common_type(0);
}

//! Binary == operator for real complex.
template <class RealType1,
          class RealType2,
          // Constraints to avoid participation in oparator==() for every possible RHS
          std::enable_if_t<std::is_convertible_v<RealType1, RealType2>, int> = 0>
KOKKOS_INLINE_FUNCTION bool operator==(RealType1 const& x, Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x) == common_type(y.real()) && common_type(0) == common_type(y.imag());
}

//! Binary != operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION bool operator!=(Complex<RealType1> const& x,
                                       Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) != common_type(y.real()) ||
           common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for std::complex complex.
template <class RealType1, class RealType2>
inline bool operator!=(std::complex<RealType1> const& x, Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) != common_type(y.real()) ||
           common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for complex std::complex.
template <class RealType1, class RealType2>
inline bool operator!=(Complex<RealType1> const& x, std::complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) != common_type(y.real()) ||
           common_type(x.imag()) != common_type(y.imag());
}

//! Binary != operator for complex real.
template <class RealType1,
          class RealType2,
          // Constraints to avoid participation in oparator==() for every possible RHS
          std::enable_if_t<std::is_convertible_v<RealType2, RealType1>, int> = 0>
KOKKOS_INLINE_FUNCTION bool operator!=(Complex<RealType1> const& x, RealType2 const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x.real()) != common_type(y) || common_type(x.imag()) != common_type(0);
}

//! Binary != operator for real complex.
template <class RealType1,
          class RealType2,
          // Constraints to avoid participation in oparator==() for every possible RHS
          std::enable_if_t<std::is_convertible_v<RealType1, RealType2>, int> = 0>
KOKKOS_INLINE_FUNCTION bool operator!=(RealType1 const& x, Complex<RealType2> const& y) noexcept {
    using common_type = std::common_type_t<RealType1, RealType2>;
    return common_type(x) != common_type(y.real()) || common_type(0) != common_type(y.imag());
}

// </editor-fold> end Equality and inequality }}}1
//==============================================================================

//! Binary + operator for complex complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator+(
    const Complex<RealType1>& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x.real() + y.real(),
                                                             x.imag() + y.imag());
}

//! Binary + operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator+(
    const Complex<RealType1>& x, const RealType2& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x.real() + y, x.imag());
}

//! Binary + operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator+(
    const RealType1& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x + y.real(), y.imag());
}

//! Unary + operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION Complex<RealType> operator+(const Complex<RealType>& x) noexcept {
    return Complex<RealType>{+x.real(), +x.imag()};
}

//! Binary - operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator-(
    const Complex<RealType1>& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x.real() - y.real(),
                                                             x.imag() - y.imag());
}

//! Binary - operator for complex scalar.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator-(
    const Complex<RealType1>& x, const RealType2& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x.real() - y, x.imag());
}

//! Binary - operator for scalar complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator-(
    const RealType1& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x - y.real(), -y.imag());
}

//! Unary - operator for complex.
template <class RealType>
KOKKOS_INLINE_FUNCTION Complex<RealType> operator-(const Complex<RealType>& x) noexcept {
    return Complex<RealType>(-x.real(), -x.imag());
}

//! Binary * operator for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator*(
    const Complex<RealType1>& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(
        x.real() * y.real() - x.imag() * y.imag(), x.real() * y.imag() + x.imag() * y.real());
}

/// \brief Binary * operator for std::complex and complex.
///
/// This needs to exist because template parameters can't be deduced when
/// conversions occur.  We could probably fix this using hidden friends patterns
///
/// This function cannot be called in a CUDA device function, because
/// std::complex's methods and nonmember functions are not marked as
/// CUDA device functions.
template <class RealType1, class RealType2>
inline Complex<std::common_type_t<RealType1, RealType2>> operator*(const std::complex<RealType1>& x,
                                                                   const Complex<RealType2>& y) {
    return Complex<std::common_type_t<RealType1, RealType2>>(
        x.real() * y.real() - x.imag() * y.imag(), x.real() * y.imag() + x.imag() * y.real());
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator*(
    const RealType1& x, const Complex<RealType2>& y) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x * y.real(), x * y.imag());
}

/// \brief Binary * operator for RealType times complex.
///
/// This function exists because the compiler doesn't know that
/// RealType and complex<RealType> commute with respect to operator*.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator*(
    const Complex<RealType1>& y, const RealType2& x) noexcept {
    return Complex<std::common_type_t<RealType1, RealType2>>(x * y.real(), x * y.imag());
}

//! Imaginary part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType imag(const Complex<RealType>& x) noexcept {
    return x.imag();
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Impl::promote_t<ArithmeticType> imag(ArithmeticType) {
    return ArithmeticType();
}

//! Real part of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType real(const Complex<RealType>& x) noexcept {
    return x.real();
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Kokkos::Impl::promote_t<ArithmeticType> real(ArithmeticType x) {
    return x;
}

//! Constructs a complex number from magnitude and phase angle
template <class T>
KOKKOS_INLINE_FUNCTION Complex<T> polar(const T& r, const T& theta = T()) {
    KOKKOS_EXPECTS(r >= 0);
    return Complex<T>(r * cos(theta), r * sin(theta));
}

//! Absolute value (magnitude) of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION RealType abs(const Complex<RealType>& x) {
    return hypot(x.real(), x.imag());
}

//! Power of a complex number
template <class T>
KOKKOS_INLINE_FUNCTION Complex<T> pow(const Complex<T>& x, const T& y) {
    T r = abs(x);
    T theta = atan2(x.imag(), x.real());
    return polar(pow(r, y), y * theta);
}

template <class T>
KOKKOS_INLINE_FUNCTION Complex<T> pow(const T& x, const Complex<T>& y) {
    return pow(Complex<T>(x), y);
}

template <class T>
KOKKOS_INLINE_FUNCTION Complex<T> pow(const Complex<T>& x, const Complex<T>& y) {
    return x == T() ? T() : exp(y * log(x));
}

template <class T, class U, class = std::enable_if_t<std::is_arithmetic_v<T>>>
KOKKOS_INLINE_FUNCTION Complex<Kokkos::Impl::promote_2_t<T, U>> pow(const T& x,
                                                                    const Complex<U>& y) {
    using type = Kokkos::Impl::promote_2_t<T, U>;
    return pow(type(x), Complex<type>(y));
}

template <class T, class U, class = std::enable_if_t<std::is_arithmetic_v<U>>>
KOKKOS_INLINE_FUNCTION Complex<Kokkos::Impl::promote_2_t<T, U>> pow(const Complex<T>& x,
                                                                    const U& y) {
    using type = Kokkos::Impl::promote_2_t<T, U>;
    return pow(Complex<type>(x), type(y));
}

template <class T, class U>
KOKKOS_INLINE_FUNCTION Complex<Kokkos::Impl::promote_2_t<T, U>> pow(const Complex<T>& x,
                                                                    const Complex<U>& y) {
    using type = Kokkos::Impl::promote_2_t<T, U>;
    return pow(Complex<type>(x), Complex<type>(y));
}

//! Square root of a complex number. This is intended to match the stdc++
//! implementation, which returns sqrt(z*z) = z; where z is complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION scaluq::Complex<RealType> sqrt(const Complex<RealType>& x) {
    RealType r = x.real();
    RealType i = x.imag();

    if (r == RealType()) {
        RealType t = sqrt(fabs(i) / 2);
        return scaluq::Complex<RealType>(t, i < RealType() ? -t : t);
    } else {
        RealType t = sqrt(2 * (abs(x) + fabs(r)));
        RealType u = t / 2;
        return r > RealType() ? scaluq::Complex<RealType>(u, i / t)
                              : scaluq::Complex<RealType>(fabs(i) / t, i < RealType() ? -u : u);
    }
}

//! Conjugate of a complex number.
template <class RealType>
KOKKOS_INLINE_FUNCTION Complex<RealType> conj(const Complex<RealType>& x) noexcept {
    return Complex<RealType>(real(x), -imag(x));
}

template <class ArithmeticType>
KOKKOS_INLINE_FUNCTION constexpr Complex<Kokkos::Impl::promote_t<ArithmeticType>> conj(
    ArithmeticType x) {
    using type = Kokkos::Impl::promote_t<ArithmeticType>;
    return Complex<type>(x, -type());
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator/(
    const Complex<RealType1>& x, const RealType2& y) noexcept(noexcept(RealType1{} / RealType2{})) {
    return Complex<std::common_type_t<RealType1, RealType2>>(real(x) / y, imag(x) / y);
}

//! Binary operator / for complex.
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator/(
    const Complex<RealType1>& x,
    const Complex<RealType2>& y) noexcept(noexcept(RealType1{} / RealType2{})) {
    // Scale (by the "1-norm" of y) to avoid unwarranted overflow.
    // If the real part is +/-Inf and the imaginary part is -/+Inf,
    // this won't change the result.
    using common_real_type = std::common_type_t<RealType1, RealType2>;
    const common_real_type s = fabs(real(y)) + fabs(imag(y));

    // If s is 0, then y is zero, so x/y == real(x)/0 + i*imag(x)/0.
    // In that case, the relation x/y == (x/s) / (y/s) doesn't hold,
    // because y/s is NaN.
    if (s == 0.0) {
        return Complex<common_real_type>(real(x) / s, imag(x) / s);
    } else {
        const Complex<common_real_type> x_scaled(real(x) / s, imag(x) / s);
        const Complex<common_real_type> y_conj_scaled(real(y) / s, -imag(y) / s);
        const RealType1 y_scaled_abs =
            real(y_conj_scaled) * real(y_conj_scaled) +
            imag(y_conj_scaled) * imag(y_conj_scaled);  // abs(y) == abs(conj(y))
        Complex<common_real_type> result = x_scaled * y_conj_scaled;
        result /= y_scaled_abs;
        return result;
    }
}

//! Binary operator / for complex and real numbers
template <class RealType1, class RealType2>
KOKKOS_INLINE_FUNCTION Complex<std::common_type_t<RealType1, RealType2>> operator/(
    const RealType1& x, const Complex<RealType2>& y) noexcept(noexcept(RealType1{} / RealType2{})) {
    return Complex<std::common_type_t<RealType1, RealType2>>(x) / y;
}

template <class RealType>
std::ostream& operator<<(std::ostream& os, const Complex<RealType>& x) {
    const std::complex<RealType> x_std(scaluq::real(x), scaluq::imag(x));
    os << x_std;
    return os;
}

template <class RealType>
std::istream& operator>>(std::istream& is, Complex<RealType>& x) {
    std::complex<RealType> x_std;
    is >> x_std;
    x = x_std;  // only assigns on success of above
    return is;
}

}  // namespace scaluq

namespace Kokkos {

template <class T>
struct reduction_identity<scaluq::Complex<T>> {
    using t_red_ident = reduction_identity<T>;
    KOKKOS_FORCEINLINE_FUNCTION constexpr static scaluq::Complex<T> sum() noexcept {
        return scaluq::Complex<T>(t_red_ident::sum(), t_red_ident::sum());
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr static scaluq::Complex<T> prod() noexcept {
        return scaluq::Complex<T>(t_red_ident::prod(), t_red_ident::sum());
    }
};

}  // namespace Kokkos

/*
#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_COMPLEX
#endif
#endif  // KOKKOS_COMPLEX_HPP
*/ // removed by Qulacs-Osaka on 2024/12/11
