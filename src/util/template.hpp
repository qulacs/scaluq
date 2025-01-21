#pragma once

#define FLOAT(Fp) template <std::floating_point Fp>
#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>
#define FLOAT_DECLARE_CLASS(Class) \
    template class Class<float>;   \
    template class Class<double>;
#ifdef __CUDACC__
#define FLOAT_AND_SPACE_DECLARE_CLASS(Class)   \
    template class Class<float, CPUSpace>;     \
    template class Class<float, DefaultSpace>; \
    template class Class<double, CPUSpace>;    \
    template class Class<double, DefaultSpace>;
#else
#define FLOAT_AND_SPACE_DECLARE_CLASS(Class)   \
    template class Class<float, DefaultSpace>; \
    template class Class<double, DefaultSpace>;
#endif

#define CALL_MACRO_FOR_FLOAT(MACRO) MACRO(float) MACRO(double)
#ifdef __CUDACC__
#define CALL_MACRO_FOR_FLOAT_AND_SPACE(MACRO) \
    MACRO(float, CPUSpace)                    \
    MACRO(double, CPUSpace) MACRO(float, DefaultSpace) MACRO(double, DefaultSpace)
#else
#define CALL_MACRO_FOR_FLOAT_AND_SPACE(MACRO) \
    MACRO(float, DefaultSpace)                \
    MACRO(double, DefaultSpace)
#endif
#define CALL_MACRO_FOR_COMPLEX(MACRO) MACRO(Complex<float>) MACRO(Complex<double>)
#define CALL_MACRO_FOR_UINT(MACRO) \
    MACRO(std::uint8_t) MACRO(std::uint16_t) MACRO(std::uint32_t) MACRO(std::uint64_t)
