## Summary

`Operator::get_expectation_value(const StateVector&)` is 5–18× slower than evaluating the same terms one `PauliOperator` at a time and adding the results. This happens with both f32 and f64. It is slowest relative to the per-term path when the operator has only a few terms.

## Measurement

The measurements use `main` at 01934038, the OpenMP build (g++-16, Release), and 16 threads on an Apple M3 Ultra. The state is a Haar random state, and the operator has two terms, `Z 0 Z 1` (coefficient 1.0) and `X 0 X 2` (coefficient 0.5). Times are in ms per call.

| precision | n | `Operator` (2 terms) | 2 × `PauliOperator` | `get_squared_norm` |
|---|---|---|---|---|
| f64 | 20 | 1.52 | 0.30 | 0.17 |
| f64 | 24 | 22.9 | 2.81 | 1.39 |
| f64 | 26 | 91.0 | 8.64 | 4.35 |
| f32 | 20 | 1.50 | 0.30 | 0.16 |
| f32 | 24 | 20.9 | 1.57 | 0.90 |
| f32 | 26 | 87.7 | 4.91 | 3.23 |

Both paths return the same value up to rounding. The same pattern appeared in an experimental SYCL build on an Apple GPU: at n = 28, the `Operator` call took 250 ms and `get_squared_norm` took 4.5 ms. So the cause does not seem specific to OpenMP.

<details><summary>Reproducer (public API only)</summary>

```cpp
auto state = StateVector<P, ExecutionSpace::Default>::Haar_random_state(n, 1);
PauliOperator<P> p1("Z 0 Z 1", 1.0), p2("X 0 X 2", 0.5);
Operator<P, ExecutionSpace::Default> op({p1, p2});
auto a = op.get_expectation_value(state);                                   // slow
auto b = p1.get_expectation_value(state) + p2.get_expectation_value(state); // fast
```

Each call is warmed up once and then timed over 5 calls, with `Kokkos::fence()` around the timed loop.
</details>

## Likely cause (not verified)

`src/operator/operator.cpp` runs one `parallel_reduce` over `MDRangePolicy<Rank<2>>({0, 0}, {nterms, dim >> 1})` with a `ComplexType` result. With few terms the iteration space is very narrow (2 × 2^(n−1)), and the default MDRange tiling may not suit that shape. Each iteration also reads the term data from `_terms[term_id]`.

## Possible directions

- Use a one-dimensional `RangePolicy` over the state for each term, as `PauliOperator::get_expectation_value` does. Or flatten the iteration space to one dimension.
- Or choose an explicit tile shape. It should be long in the state dimension.

I have not tried either yet.
