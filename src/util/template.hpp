#pragma once

#define FLOAT(Fp) template <std::floating_point Fp>
#define FLOAT_DECLARE_CLASS(Class) \
    template class Class<float>;   \
    template class Class<double>;
#define CALL_MACRO_FOR_FLOAT(MACRO) MACRO(float) MACRO(double)
#define CALL_MACRO_FOR_COMPLEX(MACRO) MACRO(Complex<float>) MACRO(Complex<double>)
#define CALL_MACRO_FOR_UINT(MACRO) \
    MACRO(std::uint8_t) MACRO(std::uint16_t) MACRO(std::uint32_t) MACRO(std::uint64_t)
