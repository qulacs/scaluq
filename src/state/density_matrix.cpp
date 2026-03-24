#include <scaluq/prec_space.hpp>
#include <scaluq/state/density_matrix.hpp>

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, Space>::DensityMatrix(std::uint64_t n_qubits)
    : _n_qubits(n_qubits),
      _dim(1ULL << n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim, this->_dim) {
    set_zero_state();
}

template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, Space>::DensityMatrix(Kokkos::View<ComplexType**, ExecutionSpaceType> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}

template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, Space>::DensityMatrix(const StateVector<Prec, Space>& other)
    : _n_qubits(other.n_qubits()),
      _dim(other.dim()),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim, this->_dim) {
    Kokkos::parallel_for(
        "initialize_from_StateVector",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) {
            _raw(i, j) = other._raw(i) * internal::conj(other._raw(j));
        });
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::set_coherence_at(std::uint64_t row_index,
                                                  std::uint64_t col_index,
                                                  StdComplex c) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, row_index, col_index), host_view());
}

template <Precision Prec, ExecutionSpace Space>
StdComplex DensityMatrix<Prec, Space>::get_coherence_at(std::uint64_t row_index,
                                                        std::uint64_t col_index) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, row_index, col_index));
    ComplexType val = host_view();
    return StdComplex(static_cast<double>(val.real()), static_cast<double>(val.imag()));
}

template <Precision Prec, ExecutionSpace Space>
[[nodiscard]] DensityMatrix<Prec, Space> DensityMatrix<Prec, Space>::Haar_random_state(
    std::uint64_t n_qubits, std::uint64_t seed) {
    auto state(DensityMatrix<Prec, Space>::uninitialized_state(n_qubits));
    state.set_Haar_random_state(seed);
    return state;
}

template <Precision Prec, ExecutionSpace Space>
[[nodiscard]] DensityMatrix<Prec, Space> DensityMatrix<Prec, Space>::uninitialized_state(
    std::uint64_t n_qubits) {
    DensityMatrix<Prec, Space> state;
    state._n_qubits = n_qubits;
    state._dim = 1ULL << n_qubits;
    state._raw = Kokkos::View<ComplexType**, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("state"), state._dim, state._dim);
    return state;
}

template class DensityMatrix<internal::Prec, internal::Space>;

}  // namespace scaluq
