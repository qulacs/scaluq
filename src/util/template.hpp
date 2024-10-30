#pragma once

#define FLOAT(Fp) template <std::floating_point Fp>
#define FLOAT_DECLARE_CLASS(Class) \
    template class Class<float>;   \
    template class Class<double>;
