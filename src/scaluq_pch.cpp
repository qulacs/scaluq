// Explicit instantiation definitions for the extern templates declared in scaluq_pch.hpp.
// SKIP_PRECOMPILE_HEADERS is set for this file so the PCH (with extern declarations) is not
// injected; instead headers are included directly here to allow the definitions below.

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <scaluq/prec_space.hpp>
#include <scaluq/types.hpp>
#include <scaluq/util/math.hpp>
#include <scaluq/util/utility.hpp>

namespace scaluq::internal {
template class Complex<Prec>;
}  // namespace scaluq::internal

template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>,
    Kokkos::HostSpace>;
template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>*,
    Kokkos::HostSpace>;
template class Kokkos::View<
    scaluq::internal::Float<scaluq::internal::Prec>*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
template class Kokkos::View<
    std::uint64_t*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;

template class Kokkos::RangePolicy<scaluq::internal::SpaceType<scaluq::internal::Space>>;

template class Kokkos::Random_XorShift64_Pool<
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
