#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <scaluq/prec_space.hpp>
#include <scaluq/type/eigen_types.hpp>
#include <scaluq/types.hpp>
#include <scaluq/util/math.hpp>
#include <scaluq/util/utility.hpp>

// Explicit instantiation declarations: suppress re-instantiation of heavy types in every TU.
// Definitions live in scaluq_pch.cpp, compiled once per library target.

namespace scaluq::internal {
extern template class Complex<Prec>;
}  // namespace scaluq::internal

extern template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
extern template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>,
    Kokkos::HostSpace>;
extern template class Kokkos::View<
    scaluq::internal::Complex<scaluq::internal::Prec>*,
    Kokkos::HostSpace>;
extern template class Kokkos::View<
    scaluq::internal::Float<scaluq::internal::Prec>*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
extern template class Kokkos::View<
    std::uint64_t*,
    scaluq::internal::SpaceType<scaluq::internal::Space>>;

extern template class Kokkos::RangePolicy<
    scaluq::internal::SpaceType<scaluq::internal::Space>>;

extern template class Kokkos::Random_XorShift64_Pool<
    scaluq::internal::SpaceType<scaluq::internal::Space>>;
