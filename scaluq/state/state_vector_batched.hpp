#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {
class StateVectorBatched {
    UINT _batch_size;
    UINT _n_qubits;
    UINT _dim;

public:
    Kokkos::View<Complex**> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(UINT batch_size, UINT n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    UINT n_qubits() const { return this->_n_qubits; }

    UINT dim() const { return this->_dim; }

    UINT batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector& state);
    void set_state_vector(UINT batch_id, const StateVector& state);
    [[nodiscard]] StateVector get_state_vector(UINT batch_id) const;

    void set_zero_state();

    [[nodiscard]] static StateVectorBatched Haar_random_states(UINT batch_size,
                                                               UINT n_qubits,
                                                               UINT seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<Complex>> amplitudes() const;

    std::vector<double> get_squared_norm() const;

    void normalize();

    // not implemented yet
    [[nodiscard]] std::vector<double> get_zero_probability(UINT target_qubit_index) const;
    // not implemented yet
    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<UINT>& measured_values) const;
    // not implemented yet
    [[nodiscard]] std::vector<double> get_entropy() const;

    void add_state_vector(const StateVectorBatched& states);
    void add_state_vector_with_coef(const Complex& coef, const StateVectorBatched& states);
    void multiply_coef(const Complex& coef);

    void load(const std::vector<std::vector<Complex>>& states);
    [[nodiscard]] StateVectorBatched copy() const;

    template <bool display_indexes = true>
    std::string to_string() const {
        std::stringstream os;
        auto states_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _raw);
        Kokkos::fence();

        os << " *** Quantum States ***\n";
        os << " * Qubit Count : " << _n_qubits << '\n';
        os << " * Dimension   : " << _dim << '\n';
        for (UINT b = 0; b < _batch_size; ++b) {
            StateVector tmp(_n_qubits);
            os << "--------------------\n";
            os << " * Batch_id    : " << b << '\n';
            os << " * State vector : \n";
            for (UINT i = 0; i < _dim; ++i) {
                if constexpr (display_indexes) {
                    os <<
                        [](UINT n, UINT len) {
                            std::string tmp;
                            while (len--) {
                                tmp += ((n >> len) & 1) + '0';
                            }
                            return tmp;
                        }(i, _n_qubits)
                       << ": ";
                }
                os << states_h(b, i) << std::endl;
            }
        }
        return os.str();
    }
};
}  // namespace scaluq
