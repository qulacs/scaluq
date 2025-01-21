#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {

FLOAT_AND_SPACE(Fp, Sp)
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVector<Fp, Sp>& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            state._raw[basis] *= diag[0];
            state._raw[basis | target_mask] *= diag[1];
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp, Sp)                         \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Fp>&, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVectorBatched<Fp, Sp>& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0},
            {states.batch_size(), states.dim() >> std::popcount(target_mask | control_mask)}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            states._raw(batch_id, basis) *= diag[0];
            states._raw(batch_id, basis | target_mask) *= diag[1];
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp, Sp)                         \
    template void one_target_diagonal_matrix_gate( \
        std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Fp>&, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

FLOAT_AND_SPACE(Fp, Sp)
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp, Sp>& mat,
                        StateVector<Fp, Sp>& state) {
    auto values = mat._values;

    Kokkos::View<Complex<Fp>*, Sp> update(Kokkos::ViewAllocateWithoutInitializing("update"),
                                          state.dim());
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state.dim()), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::View<Complex<Fp>*, Sp, Kokkos::MemoryTraits<Kokkos::Atomic>> update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0},
            {(std::int64_t)(state.dim() >> std::popcount(target_mask | control_mask)),
             (std::int64_t)values.size()}),
        KOKKOS_LAMBDA(std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            std::uint32_t src_index =
                internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            std::uint32_t dst_index =
                internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    state._raw = update;
}

FLOAT_AND_SPACE(Fp, Sp)
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp, Sp>& mat,
                        StateVectorBatched<Fp, Sp>& states) {
    auto values = mat._values;
    const std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);

    Kokkos::View<Complex<Fp>**, Kokkos::LayoutRight, Sp> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), states.batch_size(), states.dim());

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);
            }
        });
    Kokkos::fence();

    Kokkos::View<Complex<Fp>**, Kokkos::LayoutRight, Sp, Kokkos::MemoryTraits<Kokkos::Atomic>>
        update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<3>>(
            {0, 0, 0},
            {(std::int64_t)states.batch_size(),
             (std::int64_t)(states.dim() >> std::popcount(target_mask | control_mask)),
             (std::int64_t)values.size()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(batch_id, dst_index) += v * states._raw(batch_id, src_index);
        });
    Kokkos::fence();
    states._raw = update;
}

#define FUNC_MACRO(Fp, Sp)            \
    template void sparse_matrix_gate( \
        std::uint64_t, std::uint64_t, const SparseMatrix<Fp, Sp>&, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

#define FUNC_MACRO(Fp, Sp)            \
    template void sparse_matrix_gate( \
        std::uint64_t, std::uint64_t, const SparseMatrix<Fp, Sp>&, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
