## Summary

For f32 states, `get_squared_norm()` and `normalize()` lose a lot of accuracy on large states. This is worst on `ExecutionSpace::HostSerial`. `Haar_random_state` normalizes with this value, so on HostSerial a "normalized" f32 Haar state with n = 26 actually has a squared norm of 1.34.

## Measurement

The measurements use `main` at 01934038 and the OpenMP build (g++-16, Release); the Default space ran with 16 threads.

**A normalized state.** The amplitudes are random and were normalized in double on the host, then loaded with `load()`. The exact squared norm is 1.

| n | f32 HostSerial `get_squared_norm()` | f32 Default (OpenMP) `get_squared_norm()` |
|---|---|---|
| 20 | 0.99990 | 0.9999999 |
| 22 | 0.99853 | 0.999993 |
| 24 | 0.97693 | 0.99990 |
| 26 | **0.74814** | 0.99851 |

**`Haar_random_state(n, 1)`.** The table shows the squared norm of the returned state, computed in double from `get_amplitudes()`.

| n | f32 HostSerial | f32 Default (OpenMP) |
|---|---|---|
| 22 | 1.00145 | 1.000005 |
| 24 | 1.02371 | 1.000097 |
| 26 | **1.33630** | 1.00148 |

When all amplitudes are 1, the HostSerial sum stops growing at 2^24. At n = 26, `get_squared_norm()` returns 2^24 instead of 2^26, and the squared norm after `normalize()` is 4.

f64 is accurate in all of these cases.

## Cause

`StateVector::get_squared_norm` (`src/state/state_vector.cpp`) reduces into `FloatType`. On HostSerial this becomes one sequential float sum over 2^n terms, so the rounding error grows roughly linearly with the length. OpenMP splits the sum into one partial sum per thread, so its error is smaller but still grows with n. Device backends use tree-shaped reductions and are much less affected.

The same `FloatType` accumulator pattern appears in other reductions in `state_vector.cpp` and `density_matrix.cpp`, for example the probabilities and `get_trace`. I have not measured those.

## Possible directions

- Accumulate in `double` for precisions below f64 wherever device double is available, at least on the host spaces.
- Where double is not available, use a compensated (Kahan) or pairwise sum, or block the sequential host loop into chunks.
