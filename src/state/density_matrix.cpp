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
StdComplex DensityMatrix<Prec, Space>::get_coherence_at(std::uint64_t row_index,
                                                        std::uint64_t col_index) const {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, row_index, col_index));
    ComplexType val = host_view();
    return StdComplex(static_cast<double>(val.real()), static_cast<double>(val.imag()));
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
std::vector<std::vector<StdComplex>> DensityMatrix<Prec, Space>::get_matrix() const {
    Kokkos::View<ComplexType**, Kokkos::HostSpace> host_view("single_value", _dim, _dim);
    Kokkos::deep_copy(host_view, _raw);
    std::vector<std::vector<StdComplex>> matrix(_dim, std::vector<StdComplex>(_dim));
    for (std::uint64_t i = 0; i < _dim; i++) {
        for (std::uint64_t j = 0; j < _dim; j++) {
            const auto& val = host_view(i, j);
            matrix[i][j] =
                StdComplex(static_cast<double>(val.real()), static_cast<double>(val.imag()));
        }
    }
    return matrix;
}
template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, Space> DensityMatrix<Prec, Space>::copy() const {
    DensityMatrix<Prec, Space> new_state(this->_n_qubits);
    Kokkos::deep_copy(new_state._raw, this->_raw);
    return new_state;
}
template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, ExecutionSpace::Default> DensityMatrix<Prec, Space>::copy_to_default_space()
    const {
    DensityMatrix<Prec, ExecutionSpace::Default> new_state(this->_n_qubits);
    Kokkos::deep_copy(new_state._raw, this->_raw);
    return new_state;
}
template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, ExecutionSpace::Host> DensityMatrix<Prec, Space>::copy_to_host_space() const {
    DensityMatrix<Prec, ExecutionSpace::Host> new_state(this->_n_qubits);
    Kokkos::deep_copy(new_state._raw, this->_raw);
    return new_state;
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::load(const std::vector<std::vector<StdComplex>>& other) {
    if (other.size() != _dim || other[0].size() != _dim) {
        throw std::runtime_error(
            "DensityMatrix::load(const std::vector<std::vector<StdComplex>>&): Input matrix size "
            "does not match density matrix size.");
    }
    Kokkos::View<ComplexType**, Kokkos::HostSpace> host_view("host_view", _dim, _dim);

    for (std::uint64_t i = 0; i < _dim; i++) {
        for (std::uint64_t j = 0; j < _dim; j++) {
            const auto& c = other[i][j];
            host_view(i, j) =
                ComplexType(static_cast<FloatType>(c.real()), static_cast<FloatType>(c.imag()));
        }
    }
    Kokkos::deep_copy(_raw, host_view);
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::load(const DensityMatrix& other) {
    Kokkos::deep_copy(this->_raw, other._raw);
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::load(const StateVector<Prec, Space>& other) {
    if (other.n_qubits() != this->n_qubits()) {
        throw std::runtime_error(
            "DensityMatrix::load(const StateVector&): Input state vector size does not match "
            "density matrix size.");
    }
    Kokkos::parallel_for(
        "load_from_StateVector",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) {
            _raw(i, j) = other._raw(i) * internal::conj(other._raw(j));
        });
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

template <Precision Prec, ExecutionSpace Space>
[[nodiscard]] DensityMatrix<Prec, Space> DensityMatrix<Prec, Space>::Haar_random_state(
    std::uint64_t n_qubits, std::uint64_t seed) {
    auto state(DensityMatrix<Prec, Space>::uninitialized_state(n_qubits));
    state.set_Haar_random_state(seed);
    return state;
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::set_zero_state() {
    Kokkos::parallel_for(
        "set_zero_state",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) { _raw(i, j) = (i == 0 && j == 0); });
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::set_zero_norm_state() {
    Kokkos::deep_copy(_raw, 0);
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::set_computational_basis(std::uint64_t basis) {
    Kokkos::parallel_for(
        "set_computational_basis",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) {
            _raw(i, j) = (i == basis && j == basis);
        });
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::set_Haar_random_state(std::uint64_t seed) {
    auto random_pure_state = StateVector<Prec, Space>::Haar_random_state(this->_n_qubits, seed);
    this->load(random_pure_state);
}

template <Precision Prec, ExecutionSpace Space>
double DensityMatrix<Prec, Space>::get_trace() const {
    FloatType trace;
    Kokkos::parallel_reduce(
        "get_trace",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & tmp) { tmp += _raw(i, i).real(); },
        trace);
    return static_cast<double>(trace);
}

template <Precision Prec, ExecutionSpace Space>
DensityMatrix<Prec, Space> DensityMatrix<Prec, Space>::get_partial_trace(
    const std::vector<std::uint64_t>& traced_out_qubits) const {
    if (*std::ranges::max_element(traced_out_qubits) >= this->_n_qubits) {
        throw std::runtime_error(
            "DensityMatrix::get_partial_trace: Input vector for traced out qubits contains invalid "
            "qubit index.");
    }

    const std::uint64_t traced_out_mask = internal::vector_to_mask(traced_out_qubits);
    const std::uint64_t traced_out_dim = 1ULL << traced_out_qubits.size();
    const std::uint64_t remaining_dim = this->_dim / traced_out_dim;
    const std::uint64_t remaining_qubits = this->_n_qubits - traced_out_qubits.size();
    auto result =
        DensityMatrix<Prec, Space>::uninitialized_state(this->_n_qubits - traced_out_qubits.size());
    Kokkos::parallel_for(
        "get_partial_trace",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), remaining_dim * remaining_dim, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::member_type& team) {
            std::uint64_t i = team.league_rank() >> remaining_qubits;
            std::uint64_t j = team.league_rank() & (remaining_dim - 1);
            std::uint64_t row_base = internal::insert_zero_at_mask_positions(i, traced_out_mask);
            std::uint64_t col_base = internal::insert_zero_at_mask_positions(j, traced_out_mask);
            ComplexType sum = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, traced_out_dim),
                [&](std::uint64_t k, ComplexType& internal_sum) {
                    std::uint64_t row_index =
                        row_base | internal::insert_zero_at_mask_positions(k, ~traced_out_mask);
                    std::uint64_t col_index =
                        col_base | internal::insert_zero_at_mask_positions(k, ~traced_out_mask);
                    internal_sum += _raw(row_index, col_index);
                },
                sum);
            result._raw(i, j) = sum;
        });
    return result;
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::normalize() {
    const double trace = this->get_trace();
    Kokkos::parallel_for(
        "normalize",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) { _raw(i, j) /= trace; });
}

template <Precision Prec, ExecutionSpace Space>
double DensityMatrix<Prec, Space>::get_purity() const {
    FloatType purity;
    Kokkos::parallel_reduce(
        "get_purity",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j, FloatType & tmp) {
            tmp += internal::squared_norm(_raw(i, j));
        },
        purity);
    return static_cast<double>(purity);
}

template <Precision Prec, ExecutionSpace Space>
double DensityMatrix<Prec, Space>::get_zero_probability(std::uint64_t target_qubit_index) const {
    if (target_qubit_index >= this->_n_qubits) {
        throw std::runtime_error(
            "DensityMatrix::get_zero_probability: Target qubit index is out of range.");
    }
    FloatType zero_prob = 0.;
    Kokkos::parallel_reduce(
        "get_zero_probability",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim >> 1),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & internal_sum) {
            std::uint64_t basis = internal::insert_zero_to_basis_index(i, target_qubit_index);
            internal_sum += _raw(basis, basis).real();
        },
        zero_prob);
    return static_cast<double>(zero_prob);
}

template <Precision Prec, ExecutionSpace Space>
double DensityMatrix<Prec, Space>::get_marginal_probability(
    const std::vector<std::uint64_t>& measured_values) const {
    std::vector<std::uint64_t> target_index;
    std::vector<std::uint64_t> target_value;
    for (std::uint64_t i = 0; i < measured_values.size(); ++i) {
        std::uint64_t measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        } else if (measured_value != DensityMatrix<Prec, Space>::UNMEASURED) {
            throw std::runtime_error(
                "Error: DensityMatrix::get_marginal_probability(const vector<std::uint64_t>&): "
                "Invalid qubit state specified. Each qubit state must be 0, 1, or "
                "DensityMatrix::UNMEASURED.");
        }
    }

    FloatType sum = 0;
    auto d_target_index = internal::convert_vector_to_view<std::uint64_t, Space>(target_index);
    auto d_target_value = internal::convert_vector_to_view<std::uint64_t, Space>(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim >> target_index.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
            std::uint64_t basis = i;
            for (std::uint64_t cursor = 0; cursor < d_target_index.size(); cursor++) {
                std::uint64_t insert_index = d_target_index(cursor);
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis |= d_target_value(cursor) << insert_index;
            }
            lsum += _raw(basis, basis).real();
        },
        sum);

    return static_cast<double>(sum);
}

template <Precision Prec, ExecutionSpace Space>
std::vector<std::uint64_t> DensityMatrix<Prec, Space>::sampling(std::uint64_t sampling_count,
                                                                std::uint64_t seed) const {
    Kokkos::View<FloatType*, internal::SpaceType<Space>> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "sampling (compute stacked prob)",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, _dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & update, const bool final) {
            update += this->_raw(i, i).real();
            if (final) {
                stacked_prob[i + 1] = update;
            }
        });

    Kokkos::Random_XorShift64_Pool<internal::SpaceType<Space>> rand_pool(seed);
    std::vector<std::uint64_t> result(sampling_count);
    std::vector<std::uint64_t> todo(sampling_count);
    std::iota(todo.begin(), todo.end(), 0);
    while (!todo.empty()) {
        std::size_t todo_count = todo.size();
        Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> result_buf(
            Kokkos::ViewAllocateWithoutInitializing("result_buf"), todo_count);
        Kokkos::parallel_for(
            "sampling (choose)",
            Kokkos::RangePolicy<internal::SpaceType<Space>>(0, todo_count),
            KOKKOS_LAMBDA(std::uint64_t i) {
                auto rand_gen = rand_pool.get_state();
                FloatType r = static_cast<FloatType>(rand_gen.drand(0., 1.));
                std::uint64_t lo = 0, hi = stacked_prob.size();
                while (hi - lo > 1) {
                    std::uint64_t mid = (lo + hi) / 2;
                    if (stacked_prob[mid] > r) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                result_buf[i] = lo;
                rand_pool.free_state(rand_gen);
            });
        auto result_buf_host = internal::convert_view_to_vector<std::uint64_t, Space>(result_buf);
        // Especially for F16 and BF16, sampling sometimes fails with result == _dim.
        // In this case, re-sampling is performed.
        std::vector<std::uint64_t> next_todo;
        for (std::size_t i = 0; i < todo_count; i++) {
            if (result_buf_host[i] == _dim) {
                next_todo.push_back(todo[i]);
            } else {
                result[todo[i]] = result_buf_host[i];
            }
        }
        todo.swap(next_todo);
    }
    return result;
}

template <Precision Prec, ExecutionSpace Space>
double DensityMatrix<Prec, Space>::get_computational_basis_entropy() const {
    FloatType entropy = 0;
    Kokkos::parallel_reduce(
        "computational_basis_entropy",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
            FloatType prob = _raw(i, i).real();
            if (prob > 0) {
                lsum += -prob * Kokkos::log2(prob);
            }
        },
        entropy);
    return static_cast<double>(entropy);
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::add_density_matrix_with_coef(double coef,
                                                              const DensityMatrix& other) {
    Kokkos::parallel_for(
        "add_density_matrix_with_coef",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) {
            _raw(i, j) += static_cast<ComplexType>(coef) * other._raw(i, j);
        });
}

template <Precision Prec, ExecutionSpace Space>
void DensityMatrix<Prec, Space>::multiply_coef(double coef) {
    Kokkos::parallel_for(
        "multiply_coef",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {this->_dim, this->_dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, std::uint64_t j) {
            _raw(i, j) *= static_cast<ComplexType>(coef);
        });
}

template <Precision Prec, ExecutionSpace Space>
std::string DensityMatrix<Prec, Space>::to_string() const {
    std::stringstream os;
    auto matrix = this->get_matrix();
    os << "Qubit Count : " << _n_qubits << '\n';
    os << "Dimension : " << _dim << '\n';
    os << "Density Matrix : \n";
    auto binary = [](std::uint64_t n, std::uint64_t len) {
        std::string tmp;
        while (len--) {
            tmp += ((n >> len) & 1) + '0';
        }
        return tmp;
    };
    for (std::uint64_t i = 0; i < _dim; ++i) {
        auto i_binary = binary(i, _n_qubits);
        for (std::uint64_t j = 0; j < _dim; ++j) {
            os << "  (" << i_binary << ", " << binary(j, _n_qubits) << ") : " << matrix[i][j]
               << std::endl;
        }
    }
    return os.str();
}

template class DensityMatrix<internal::Prec, internal::Space>;

}  // namespace scaluq
