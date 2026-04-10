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

    template <Precision P, ExecutionSpace S>
    friend class DensityMatrix;

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
     * @attention Very slow. You should use get_matrix() instead if you can.
     */
    [[nodiscard]] StdComplex get_coherence_at(std::uint64_t row_index,
                                              std::uint64_t col_index) const;

    /**
     * @attention Very slow. You should use load() instead if you can.
     * @note is_hermitian is set to false unless diagonal element and real value is passed in.
     */
    void set_coherence_at(std::uint64_t row_index, std::uint64_t col_index, StdComplex c);

    [[nodiscard]] ComplexMatrix get_matrix() const;
    [[nodiscard]] DensityMatrix copy() const;
    [[nodiscard]] DensityMatrix<Prec, ExecutionSpace::Default> copy_to_default_space() const;
    [[nodiscard]] DensityMatrix<Prec, ExecutionSpace::Host> copy_to_host_space() const;

    void load(const ComplexMatrix& other, bool is_hermitian = false);
    void load(const DensityMatrix& other);
    void load(const StateVector<Prec, Space>& other);

    [[nodiscard]] static DensityMatrix uninitialized_state(std::uint64_t n_qubits);

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
        j = Json{{"n_qubits", state._n_qubits},
                 {"is_hermitian", state._is_hermitian},
                 {"matrix", state.get_matrix()}};
    }
    friend void from_json(const Json& j, DensityMatrix& state) {
        std::uint64_t n_qubits = j.at("n_qubits").get<std::uint64_t>();
        bool is_hermitian = j.at("is_hermitian").get<bool>();
        auto matrix = j.at("matrix").get<ComplexMatrix>();
        state = DensityMatrix::uninitialized_state(n_qubits);
        state.load(matrix, is_hermitian);
    }
};

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_state_density_matrix_hpp(nb::module_& m) {
    nb::class_<DensityMatrix<Prec, Space>>(
        m,
        "DensityMatrix",
        DocString()
            .desc("DensityMatrix representation of quantum state.")
            .desc("Qubit index is start from 0. If the (i,j)-th value of the matrix is $a_{ij}$, "
                  "the state is $\\sum_{i,j} a_{ij} \\ket{i}\\bra{j}$.")
            .build_as_google_style()
            .c_str())
        .def(nb::init<std::uint64_t>(),
             "n_qubits"_a,
             DocString()
                 .desc("Construct with specified number of qubits.")
                 .desc("Matrix is initialized with computational "
                       "basis $\\ket{0\\dots0}\\bra{0\\dots0}$.")
                 .arg("n_qubits", "int", "number of qubits")
                 .ex(DocString::Code({">>> state = DensityMatrix(2)",
                                      ">>> print(state)",
                                      "Qubit Count : 2",
                                      "Dimension : 4",
                                      "Is Hermitian : true",
                                      "Density Matrix : ",
                                      "  (00, 00) : (1,0)",
                                      "  (00, 01) : (0,0)",
                                      "  (00, 10) : (0,0)",
                                      "  (00, 11) : (0,0)",
                                      "  (01, 00) : (0,0)",
                                      "  (01, 01) : (0,0)",
                                      "  (01, 10) : (0,0)",
                                      "  (01, 11) : (0,0)",
                                      "  (10, 00) : (0,0)",
                                      "  (10, 01) : (0,0)",
                                      "  (10, 10) : (0,0)",
                                      "  (10, 11) : (0,0)",
                                      "  (11, 00) : (0,0)",
                                      "  (11, 01) : (0,0)",
                                      "  (11, 10) : (0,0)",
                                      "  (11, 11) : (0,0)",
                                      "<BLANKLINE>"}))
                 .build_as_google_style()
                 .c_str())
        .def(
            nb::init<StateVector<Prec, Space>>(),
            "state_vector"_a,
            DocString()
                .desc("Construct from state vector. The density matrix is initialized as "
                      "$\\ket{\\psi}\\bra{\\psi}$ where $\\ket{\\psi}$ is the input state vector.")
                .arg(
                    "state_vector", "StateVector", "state vector to be converted to density matrix")
                .ex(DocString::Code({">>> sv = StateVector(2)",
                                     ">>> sv.set_computational_basis(1)",
                                     ">>> dm = DensityMatrix(sv)",
                                     ">>> print(dm.get_matrix())",
                                     "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 1.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                .build_as_google_style()
                .c_str())
        .def("n_qubits",
             &DensityMatrix<Prec, Space>::n_qubits,
             DocString()
                 .desc("Get the number of qubits.")
                 .ret("int", "number of qubits")
                 .ex(DocString::Code{">>> state = DensityMatrix(3)", ">>> state.n_qubits()", "3"})
                 .build_as_google_style()
                 .c_str())
        .def("dim",
             &DensityMatrix<Prec, Space>::dim,
             DocString()
                 .desc("Get the dimension of the density matrix. ($=2^\\mathrm{n\\_qubits}$).")
                 .ret("int", "dimension of the density matrix")
                 .ex(DocString::Code{">>> state = DensityMatrix(2)", ">>> state.dim()", "4"})
                 .build_as_google_style()
                 .c_str())
        .def("is_hermitian",
             &DensityMatrix<Prec, Space>::is_hermitian,
             DocString()
                 .desc("Check if the density matrix is guaranteed to be Hermitian.")
                 .ret("bool",
                      "True if the density matrix is Hermitian, False if it may not be Hermitian")
                 .ex(DocString::Code{
                     ">>> state = DensityMatrix(2)", ">>> state.is_hermitian()", "True"})
                 .build_as_google_style()
                 .c_str())
        .def("force_hermitian",
             &DensityMatrix<Prec, Space>::force_hermitian,
             DocString()
                 .desc("Force the density matrix to be treated as Hermitian. This may enable "
                       "certain optimizations but should only be used if you are sure the matrix "
                       "is actually Hermitian.")
                 .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                     ">>> state.multiply_coef(1j)",
                                     ">>> state.multiply_coef(-1j)",
                                     ">>> state.is_hermitian()",
                                     "False",
                                     ">>> state.force_hermitian()",
                                     ">>> state.is_hermitian()",
                                     "True"})
                 .build_as_google_style()
                 .c_str())
        .def("get_coherence_at",
             &DensityMatrix<Prec, Space>::get_coherence_at,
             "row_index"_a,
             "col_index"_a,
             DocString()
                 .desc("Get coherence at specified position.")
                 .desc("This is a very slow operation. Use get_matrix() instead if possible.")
                 .arg("row_index", "int", "row index")
                 .arg("col_index", "int", "column index")
                 .ret("complex", "coherence at the specified position")
                 .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                     ">>> state.get_coherence_at(0, 0)",
                                     "(1+0j)",
                                     ">>> state.get_coherence_at(0, 1)",
                                     "0j"})
                 .build_as_google_style()
                 .c_str())
        .def("set_coherence_at",
             &DensityMatrix<Prec, Space>::set_coherence_at,
             "row_index"_a,
             "col_index"_a,
             "value"_a,
             DocString()
                 .desc("Set coherence at specified position.")
                 .desc("This is a very slow operation. Use load() instead if possible.")
                 .desc("is_hermitian is set to False unless diagonal element and real value is "
                       "passed in.")
                 .arg("row_index", "int", "row index")
                 .arg("col_index", "int", "column index")
                 .arg("value", "complex", "value to set at the specified position")
                 .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                     ">>> state.get_matrix()",
                                     "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]",
                                     ">>> state.set_coherence_at(0, 1, 0.5+0.5j)",
                                     ">>> state.get_matrix()",
                                     "[[1.+0.j 0.5+0.5j 0.+0.j 0.+0.j]",
                                     " [0.5-0.5j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"})
                 .build_as_google_style()
                 .c_str())
        .def("get_matrix",
             &DensityMatrix<Prec, Space>::get_matrix,
             DocString()
                 .desc("Get the density matrix as ndarray.")
                 .ret("Annotated[numpy.typing.NDArray[numpy.complex128], dict(shape=(None, None), "
                      "order=’C’)]",
                      "density matrix as 2D array")
                 .ex(DocString::Code({">>> state = DensityMatrix(2)",
                                      ">>> print(state.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("copy",
             &DensityMatrix<Prec, Space>::copy,
             DocString()
                 .desc("Get a copy of the density matrix.")
                 .ret("DensityMatrix", "a copy of the density matrix")
                 .ex(DocString::Code({">>> state1 = DensityMatrix(2)",
                                      ">>> state2 = state1.copy()",
                                      ">>> print(state1.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]",
                                      ">>> print(state2.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("copy_to_default_space",
             &DensityMatrix<Prec, Space>::copy_to_default_space,
             DocString()
                 .desc("Get a copy of the density matrix in the default execution space.")
                 .ret(std::string(":class:`scaluq.default.") + to_string(Prec) + ".DensityMatrix`",
                      "a copy of the density matrix in the default execution space")
                 .ex(DocString::Code({">>> state_host = DensityMatrix(2)",
                                      ">>> state_default = state_host.copy_to_default_space()",
                                      ">>> print(state_default.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("copy_to_host_space",
             &DensityMatrix<Prec, Space>::copy_to_host_space,
             DocString()
                 .desc("Get a copy of the density matrix in the host execution space.")
                 .ret(std::string(":class:`scaluq.host.") + to_string(Prec) + ".DensityMatrix`",
                      "a copy of the density matrix in the host execution space")
                 .ex(DocString::Code({">>> state_default = DensityMatrix(2)",
                                      ">>> state_host = state_default.copy_to_host_space()",
                                      ">>> print(state_host.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("load",
             nb::overload_cast<const ComplexMatrix&, bool>(&DensityMatrix<Prec, Space>::load),
             "matrix"_a,
             "is_hermitian"_a = false,
             DocString()
                 .desc("Load the density matrix from a 2D array.")
                 .arg("matrix",
                      "Annotated[numpy.typing.NDArray[numpy.complex128], dict(shape=(None, None), "
                      "order=’C’)]",
                      "2D array representing the density matrix")
                 .arg("is_hermitian",
                      "bool",
                      "Whether the input matrix is guaranteed to be Hermitian.")
                 .ex(DocString::Code({"import numpy as np",
                                      ">>> state = DensityMatrix(2)",
                                      ">>> matrix = np.array([[0.5+0.5j, 0], [0, 0.5-0.5j]])",
                                      ">>> state.load(matrix, is_hermitian=True)",
                                      ">>> print(state.get_matrix())",
                                      "[[0.5+0.5j 0.+0.j]",
                                      " [0.+0.j 0.5-0.5j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def(
            "load",
            nb::overload_cast<const DensityMatrix<Prec, Space>&>(&DensityMatrix<Prec, Space>::load),
            "other"_a,
            DocString()
                .desc("Load the density matrix from another DensityMatrix.")
                .arg("other", "DensityMatrix", "DensityMatrix to load from")
                .ex(DocString::Code(
                    {">>> state1 = DensityMatrix.Haar_random_state(1)",
                     ">>> print(state1.get_matrix()) # doctest: +SKIP",
                     "[[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]",
                     " [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]",
                     ">>> state2 = DensityMatrix(1)",
                     ">>> state2.load(state1)",
                     ">>> print(state2.get_matrix()) # doctest: +SKIP",
                     "[[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]",
                     " [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]"}))
                .build_as_google_style()
                .c_str())
        .def("load",
             nb::overload_cast<const StateVector<Prec, Space>&>(&DensityMatrix<Prec, Space>::load),
             "state_vector"_a,
             DocString()
                 .desc("Load the density matrix from a StateVector. The density matrix is set to "
                       "$\\ket{\\psi}\\bra{\\psi}$ where $\\ket{\\psi}$ is the input state vector.")
                 .arg("state_vector", "StateVector", "StateVector to load from")
                 .ex(DocString::Code({">>> sv = StateVector(2)",
                                      ">>> sv.set_computational_basis(1)",
                                      ">>> dm = DensityMatrix(2)",
                                      ">>> dm.load(sv)",
                                      ">>> print(dm.get_matrix())",
                                      "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 1.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def_static(
            "uninitialized_state",
            &DensityMatrix<Prec, Space>::uninitialized_state,
            "n_qubits"_a,
            DocString()
                .desc("Create an uninitialized density matrix with specified number of qubits.")
                .arg("n_qubits", "int", "number of qubits")
                .ret("DensityMatrix", "an uninitialized density matrix")
                .build_as_google_style()
                .c_str())
        .def_static(
            "Haar_random_state",
            [](std::uint64_t n_qubits, std::optional<std::uint64_t> seed) {
                return DensityMatrix<Prec, Space>::Haar_random_state(
                    n_qubits, seed.value_or(std::random_device()()));
            },
            "n_qubits"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Create a density matrix representing a Haar random state.")
                .arg("n_qubits", "int", "number of qubits")
                .arg("seed",
                     "int | None",
                     true,
                     "random seed",
                     "If not specified, the value from random device is used.")
                .ret("DensityMatrix", "a density matrix representing a Haar random state")
                .ex(DocString::Code(
                    {">>> state = DensityMatrix.Haar_random_state(1) # doctest: +SKIP",
                     ">>> print(state.get_matrix())",
                     "[[ 0.35920411+8.30722924e-18j -0.29205502-3.80631554e-01j]",
                     " [-0.29205502+3.80631554e-01j  0.64079589+5.48595618e-18j]]",
                     ">>> state1 = DensityMatrix.Haar_random_state(1, seed=42)",
                     ">>> state2 = DensityMatrix.Haar_random_state(1, seed=42)",
                     ">>> print(state1.get_matrix()) # doctest: +SKIP",
                     "[[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]",
                     " [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]",
                     ">>> print(state2.get_matrix()) # doctest: +SKIP",
                     "[[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]",
                     " [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]"}))
                .build_as_google_style()
                .c_str())
        .def("set_zero_state",
             &DensityMatrix<Prec, Space>::set_zero_state,
             DocString()
                 .desc("Set the density matrix to the zero state $\\ket{0\\dots0}\\bra{0\\dots0}$.")
                 .ex(DocString::Code({">>> state = DensityMatrix.uninitialized_state(2)",
                                      ">>> state.set_zero_state()",
                                      ">>> print(state.get_matrix())",
                                      "[[1.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("set_zero_norm_state",
             &DensityMatrix<Prec, Space>::set_zero_norm_state,
             DocString()
                 .desc("Set the density matrix to the 0 (zero matrix).")
                 .ex(DocString::Code({">>> state = DensityMatrix.uninitialized_state(2)",
                                      ">>> state.set_zero_norm_state()",
                                      ">>> print(state.get_matrix())",
                                      "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                                      " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def(
            "set_computational_basis",
            &DensityMatrix<Prec, Space>::set_computational_basis,
            "basis_index"_a,
            DocString()
                .desc("Set the density matrix to a computational basis state $\\ket{b}\\bra{b}$ "
                      "where $b$ is the binary representation of the input basis index.")
                .arg(
                    "basis_index", "int", "index of the computational basis state to set (0-based)")
                .ex(DocString::Code(
                    {">>> state = DensityMatrix.uninitialized_state(2)",
                     ">>> state.set_computational_basis(2) # sets state to |10><10|",
                     ">>> print(state.get_matrix())",
                     "[[0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]",
                     " [0.+0.j 0.+0.j 1.+0.j 0.+0.j]",
                     " [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]"}))
                .build_as_google_style()
                .c_str())
        .def(
            "set_Haar_random_state",
            [](std::optional<std::uint64_t> seed, DensityMatrix<Prec, Space>& self) {
                self.set_Haar_random_state(seed.value_or(std::random_device{}()));
            },
            "seed"_a = std::nullopt,
            DocString()
                .desc("Set the density matrix to represent a Haar random state.")
                .arg("seed",
                     "int | None",
                     true,
                     "random seed",
                     "If not specified, the value from random device is used.")
                .ex(DocString::Code(
                    {">>> state = DensityMatrix(1)",
                     ">>> state.set_Haar_random_state(seed=42)",
                     ">>> print(state.get_matrix()) # doctest: +SKIP",
                     "[[ 0.7662367 -2.34701458e-17j -0.35360255-2.32558070e-01j]",
                     " [-0.35360255+2.32558070e-01j  0.2337633 -7.47914693e-19j]]"}))
                .build_as_google_style()
                .c_str())
        .def("get_trace",
             &DensityMatrix<Prec, Space>::get_trace,
             DocString()
                 .desc("Calculate the trace of the density matrix.")
                 .ret("complex", "trace of the density matrix")
                 .ex(DocString::Code{
                     ">>> state = DensityMatrix(2)", ">>> state.get_trace()", "(1+0j)"})
                 .build_as_google_style()
                 .c_str())
        .def("get_partial_trace",
             &DensityMatrix<Prec, Space>::get_partial_trace,
             "traced_out_qubits"_a,
             DocString()
                 .desc("Calculate the partial trace of the density matrix by tracing out the "
                       "specified qubits.")
                 .arg("traced_out_qubits",
                      "collections.abc.Sequence[int]",
                      "indices of qubits to be traced out")
                 .ret("DensityMatrix", "the resulting density matrix after partial trace")
                 .ex(DocString::Code(
                     {">>> state = DensityMatrix.Haar_random_state(2)",
                      ">>> print(state.get_matrix()) # doctest: +SKIP",
                      "[[ 0.51491987-2.74660016e-17j -0.23762498-1.56281696e-01j",
                      "   0.10531599+3.49755601e-01j -0.13503799+1.31271109e-01j]",
                      " [-0.23762498+1.56281696e-01j  0.15709162+1.84452504e-20j",
                      "  -0.15475438-1.29440927e-01j  0.02247559-1.01563879e-01j]",
                      " [ 0.10531599-3.49755601e-01j -0.15475438+1.29440927e-01j",
                      "   0.25910912-1.49078030e-18j  0.06154578+1.18572309e-01j]",
                      " [-0.13503799-1.31271109e-01j  0.02247559+1.01563879e-01j",
                      "   0.06154578-1.18572309e-01j  0.06887938-1.44012992e-19j]]",
                      ">>> reduced_state = state.get_partial_trace([1])",
                      ">>> print(reduced_state.get_matrix()) # doctest: +SKIP",
                      "[[ 0.774029  -2.89567819e-17j -0.17607919-3.77093869e-02j]",
                      " [-0.17607919+3.77093869e-02j  0.225971  -1.25567742e-19j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("normalize",
             &DensityMatrix<Prec, Space>::normalize,
             DocString()
                 .desc("Normalize the density matrix so that its trace becomes 1.")
                 .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                     ">>> state.multiply_coef(2)",
                                     ">>> state.normalize()",
                                     ">>> print(state.get_trace())",
                                     "(1+0j)"})
                 .build_as_google_style()
                 .c_str())
        .def("get_purity",
             &DensityMatrix<Prec, Space>::get_purity,
             DocString()
                 .desc("Calculate the purity of the quantum state represented by the density "
                       "matrix. Purity is defined as $\\mathrm{Tr}(\\rho^2)$ where $\\rho$ is the "
                       "density matrix.")
                 .note("The matrix must be hermitian and normalized")
                 .ret("float", "purity of the quantum state")
                 .ex(DocString::Code{">>> state1 = DensityMatrix.Haar_random_state(2, 0)",
                                     ">>> print(state1.get_purity()) # doctest: +SKIP",
                                     "1.0000000000000002",
                                     ">>> state2 = DensityMatrix.Haar_random_state(2, 1)",
                                     ">>> state1.multiply_coef(0.5)",
                                     ">>> state1.add_density_matrix_with_coef(0.5, state2)",
                                     ">>> print(state1.get_purity()) # doctest: +SKIP",
                                     "0.6238584782007198"})
                 .build_as_google_style()
                 .c_str())
        .def(
            "get_zero_probability",
            &DensityMatrix<Prec, Space>::get_zero_probability,
            "target_qubit_index"_a,
            DocString()
                .desc("Calculate the probability of measuring 0 on the target qubit.")
                .arg("target_qubit_index", "int", "index of the target qubit")
                .note("The matrix must be hermitian and normalized")
                .ret("float", "probability of measuring 0 on the target qubit")
                .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                    ">>> state.set_computational_basis(2) # sets state to |10><10|",
                                    ">>> state.get_zero_probability(0)",
                                    "0.0",
                                    ">>> state.get_zero_probability(1)",
                                    "1.0"})
                .build_as_google_style()
                .c_str())
        .def(
            "get_marginal_probability",
            &DensityMatrix<Prec, Space>::get_marginal_probability,
            "measured_values"_a,
            DocString()
                .desc("Get the marginal probability to observe as given.")
                .note("The matrix must be hermitian and normalized")
                .arg("measured_values",
                     "list[int]",
                     "list with len n_qubits.",
                     "`0`, `1` or :attr:`.UNMEASURED` is allowed for each elements. `0` or `1` "
                     "shows the qubit is observed and the value is got. :attr:`.UNMEASURED` "
                     "shows the the qubit is not observed.")
                .ret("float", "probability to observe as given")
                .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                    ">>> state.set_computational_basis(2) # sets state to |10><10|",
                                    ">>> state.get_marginal_probability([0, 1])",
                                    "1.0",
                                    ">>> state.get_marginal_probability([0, UNMEASURED])",
                                    "1.0",
                                    ">>> state.get_marginal_probability([UNMEASURED, 1])",
                                    "1.0",
                                    ">>> state.get_marginal_probability([1, UNMEASURED])",
                                    "0.0"})
                .build_as_google_style()
                .c_str())
        .def(
            "sampling",
            [](const DensityMatrix<Prec, Space>& self,
               std::uint64_t sampling_count,
               std::optional<std::uint64_t> seed) {
                return self.sampling(sampling_count, seed.value_or(std::random_device{}()));
            },
            "sampling_count"_a,
            "seed"_a = std::nullopt,
            DocString()
                .desc("Sampling density matrix independently and get list of computational basis")
                .note("The matrix must be hermitian and normalized")
                .arg("sampling_count", "int", "number of samples to draw")
                .arg("seed",
                     "int | None",
                     true,
                     "random seed",
                     "If not specified, the value from random device is used.")
                .ret("List[int]", "list of sampled computational basis represented as integers")
                .ex(DocString::Code({">>> import numpy as np",
                                     ">>> state = DensityMatrix.uninitialized_state(1)",
                                     ">>> state.load(np.array([[0.5, -0.5j], [0.5j, 0.5]]), True)",
                                     ">>> print(state.sampling(10, seed=42)) # doctest: +SKIP",
                                     "[0, 1, 1, 0, 0, 0, 1, 0, 1, 1]"}))
                .build_as_google_style()
                .c_str())
        .def(
            "get_computational_basis_entropy",
            &DensityMatrix<Prec, Space>::get_computational_basis_entropy,
            DocString()
                .desc("Calculate the computational basis entropy of the quantum state represented "
                      "by the density matrix. Computational basis entropy is defined as $-\\sum_i "
                      "p_i \\log_2 p_i$ where $p_i$ is the probability of measuring the "
                      "computational basis state $\\ket{i}$, which can be calculated as the "
                      "(i,i)-th element of the density matrix.")
                .note("The matrix must be hermitian and normalized")
                .ret("float", "computational basis entropy of the quantum state")
                .ex(DocString::Code{">>> state = DensityMatrix(2)",
                                    ">>> state.set_computational_basis(2) # sets state to |10><10|",
                                    ">>> state.get_computational_basis_entropy()",
                                    ">>> import numpy as np",
                                    ">>> state = DensityMatrix.uninitialized_state(1)",
                                    ">>> state.load(np.array([[0.5, -0.5j], [0.5j, 0.5]]), True)",
                                    ">>> state.get_computational_basis_entropy()",
                                    "1.0"})
                .build_as_google_style()
                .c_str())
        .def("add_density_matrix_with_coef",
             &DensityMatrix<Prec, Space>::add_density_matrix_with_coef,
             "coef"_a,
             "other"_a,
             DocString()
                 .desc("Add another density matrix to this density matrix with a coefficient. This "
                       "performs the operation $\\rho \\leftarrow \\rho + c \\sigma$ where $\\rho$ "
                       "is this density matrix, $c$ is the coefficient and $\\sigma$ is the other "
                       "density matrix.")
                 .arg("coef", "complex", "coefficient to multiply the other density matrix")
                 .arg("other", "DensityMatrix", "the other density matrix to be added")
                 .ex(DocString::Code(
                     {">>> state1 = DensityMatrix.Haar_random_state(1, 1)",
                      ">>> state2 = DensityMatrix.Haar_random_state(1, 2)",
                      ">>> print(state1.get_matrix()) # doctest: +SKIP",
                      "[[ 0.51823805+2.35020398e-18j -0.20611927-4.55172737e-01j]",
                      " [-0.20611927+4.55172737e-01j  0.48176195+2.36023838e-19j]]",
                      ">>> print(state2.get_matrix()) # doctest: +SKIP",
                      "[[0.20777367+3.60776697e-18j 0.36552629-1.76051983e-01j]",
                      " [0.36552629+1.76051983e-01j 0.79222633-1.66856281e-17j]]",
                      ">>> state1.add_density_matrix_with_coef(0.5, state2)",
                      ">>> print(state1.get_matrix()) # doctest: +SKIP",
                      "[[ 0.62212489+4.15408747e-18j -0.02335612-5.43198728e-01j]",
                      " [-0.02335612+5.43198728e-01j  0.87787511-8.10679023e-18j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("multiply_coef",
             &DensityMatrix<Prec, Space>::multiply_coef,
             "coef"_a,
             DocString()
                 .desc("Multiply this density matrix by a coefficient. This performs the operation "
                       "$\\rho \\leftarrow c \\rho$ where $\\rho$ is this density matrix and $c$ "
                       "is the coefficient.")
                 .arg("coef", "complex", "coefficient to multiply the density matrix")
                 .ex(DocString::Code(
                     {">>> state = DensityMatrix.Haar_random_state(1, 1)",
                      ">>> print(state.get_matrix()) # doctest: +SKIP",
                      "[[ 0.51823805+2.35020398e-18j -0.20611927-4.55172737e-01j]",
                      " [-0.20611927+4.55172737e-01j  0.48176195+2.36023838e-19j]]",
                      ">>> state.multiply_coef(2)",
                      ">>> print(state.get_matrix()) # doctest: +SKIP",
                      "[[ 1.0364761 +4.70040797e-18j -0.41223854-9.10345474e-01j]",
                      " [-0.41223854+9.10345474e-01j  0.9635239 +4.72047677e-19j]]"}))
                 .build_as_google_style()
                 .c_str())
        .def("to_string",
             &DensityMatrix<Prec, Space>::to_string,
             DocString()
                 .desc("Convert the density matrix to a string representation.")
                 .build_as_google_style()
                 .c_str())
        .def("__str__",
             &DensityMatrix<Prec, Space>::to_string,
             DocString()
                 .desc("Information as `str`.")
                 .desc("Same as :meth:`.to_string()`")
                 .build_as_google_style()
                 .c_str())
        .def(
            "to_json",
            [](const DensityMatrix<Prec, Space>& state) { return Json(state).dump(); },
            DocString()
                .desc("Information as json style.")
                .ret("str", "information as json style")
                .ex(DocString::Code(
                    {">>> state = DensityMatrix(2)",
                     ">>> print(state.to_json())",
                     R"({"is_hermitian":true,"matrix":[[{"imag":0.0,"real":1.0},{"imag":0.0,"real":0.0},)"
                     R"({"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],[{"imag":0.0,"real":0.0},)"
                     R"({"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}],)"
                     R"([{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},)"
                     R"({"imag":0.0,"real":0.0}],[{"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0},)"
                     R"({"imag":0.0,"real":0.0},{"imag":0.0,"real":0.0}]],"n_qubits":2})"}))
                .build_as_google_style()
                .c_str())
        .def(
            "load_json",
            [](DensityMatrix<Prec, Space>& state, const std::string& str) {
                state = nlohmann::json::parse(str);
            },
            "json_str"_a,
            DocString()
                .desc("Read an object from the JSON representation of the density matrix.")
                .build_as_google_style()
                .c_str());
}
}  // namespace internal
#endif  // SCALUQ_USE_NANOBIND
}  // namespace scaluq
