#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class DensityMatrix {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    using FloatType = internal::Float<Prec>;
    using ComplexType = internal::Complex<Prec>;
    using ExecutionSpaceType = internal::SpaceType<Space>;

public:
    Kokkos::View<ComplexType**, ExecutionSpaceType> _raw;
    DensityMatrix() = default;
    DensityMatrix(std::uint64_t n_qubits);
    DensityMatrix(Kokkos::View<ComplexType**, ExecutionSpaceType> view);
    DensityMatrix(const StateVector<Prec, Space>& other);
    DensityMatrix(const DensityMatrix& other) = default;

    DensityMatrix& operator=(const DensityMatrix& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_coherence_at(std::uint64_t row_index, std::uint64_t col_index, StdComplex c);

    /**
     * @attention Very slow. You should use get_coherences() instead if you can.
     */
    [[nodiscard]] StdComplex get_coherence_at(std::uint64_t row_index, std::uint64_t col_index);

    [[nodiscard]] static DensityMatrix Haar_random_state(
        std::uint64_t n_qubits, std::uint64_t seed = std::random_device()());
    [[nodiscard]] static DensityMatrix uninitialized_state(std::uint64_t n_qubits);

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);
    void set_Haar_random_state(std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::vector<StdComplex> get_amplitudes() const;

    [[nodiscard]] double get_squared_norm() const;

    void normalize();
};
}  // namespace scaluq
