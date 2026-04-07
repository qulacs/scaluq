#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
class DensityMatrix {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    bool _is_hermitian;
    using FloatType = internal::Float<Prec>;
    using ComplexType = internal::Complex<Prec>;
    using ExecutionSpaceType = internal::SpaceType<Space>;

public:
    static constexpr std::uint64_t UNMEASURED = 2;
    Kokkos::View<ComplexType**, ExecutionSpaceType> _raw;
    DensityMatrix() = default;
    DensityMatrix(std::uint64_t n_qubits);
    DensityMatrix(Kokkos::View<ComplexType**, ExecutionSpaceType> view, bool is_hermitian = false);
    DensityMatrix(const StateVector<Prec, Space>& other);
    DensityMatrix(const DensityMatrix& other) = default;

    DensityMatrix& operator=(const DensityMatrix& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] bool is_hermitian() const { return this->_is_hermitian; }
    void force_hermitian() { this->_is_hermitian = true; }

    /**
     * @attention Very slow. You should use get_coherences() instead if you can.
     */
    [[nodiscard]] StdComplex get_coherence_at(std::uint64_t row_index,
                                              std::uint64_t col_index) const;

    /**
     * @attention Very slow. You should use load() instead if you can.
     * @note is_hermitian is set to false unless diagonal element and real value is passed in.
     */
    void set_coherence_at(std::uint64_t row_index, std::uint64_t col_index, StdComplex c);

    [[nodiscard]] std::vector<std::vector<StdComplex>> get_matrix() const;
    [[nodiscard]] DensityMatrix copy() const;
    [[nodiscard]] DensityMatrix<Prec, ExecutionSpace::Default> copy_to_default_space() const;
    [[nodiscard]] DensityMatrix<Prec, ExecutionSpace::Host> copy_to_host_space() const;

    void load(const std::vector<std::vector<StdComplex>>& other, bool is_hermitian = false);
    void load(const DensityMatrix& other);
    void load(const StateVector<Prec, Space>& other);

    [[nodiscard]] static DensityMatrix uninitialized_state(std::uint64_t n_qubits,
                                                           bool is_hermitian = false);

    [[nodiscard]] static DensityMatrix Haar_random_state(
        std::uint64_t n_qubits, std::uint64_t seed = std::random_device()());

    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);
    void set_Haar_random_state(std::uint64_t seed = std::random_device()());

    [[nodiscard]] StdComplex get_trace() const;
    [[nodiscard]] DensityMatrix get_partial_trace(
        const std::vector<std::uint64_t>& traced_out_qubits) const;
    void normalize();

    [[nodiscard]] double get_purity() const;

    [[nodiscard]] double get_zero_probability(std::uint64_t target_qubit_index) const;
    [[nodiscard]] double get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;

    [[nodiscard]] std::vector<std::uint64_t> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    [[nodiscard]] double get_computational_basis_entropy() const;

    void add_density_matrix_with_coef(StdComplex coef, const DensityMatrix& other);
    void multiply_coef(StdComplex coef);

    [[nodiscard]] std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const DensityMatrix& state) {
        os << state.to_string();
        return os;
    }

    friend void to_json(Json& j, const DensityMatrix& state) {
        j = Json{{"n_qubits", state._n_qubits}, {"matrix", state.get_matrix()}};
    }
    friend void from_json(const Json& j, DensityMatrix& state) {
        std::uint64_t n_qubits = j.at("n_qubits").get<std::uint64_t>();
        auto matrix = j.at("matrix").get<std::vector<std::vector<StdComplex>>>();
        state = DensityMatrix::uninitialized_state(n_qubits);
        state.load(matrix);
    }
};
}  // namespace scaluq
