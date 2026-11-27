## Summary

`__acpp_sscp_get_num_subgroups` has no Metal implementation. As a result, `sub_group::get_group_range()` (for example `it.get_sub_group().get_group_range()`) fails at JIT time with an unresolved builtin. `src/libkernel/sscp/metal/subgroup.cpp` on current `develop` does not define it.

**Environment:** AdaptiveCpp `develop` at 6c923f68 (the current tip), generic SSCP compiler, Metal backend (`-DWITH_METAL_BACKEND=ON`), Homebrew LLVM 21.1.8, macOS 26.7.1, Apple M3 Ultra.

## Fix used locally

This is appended to `src/libkernel/sscp/metal/subgroup.cpp`. With it, sub-group and group algorithm tests that use the number of sub-groups pass on an M3 Ultra.

```cpp
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  const __acpp_uint32 sg = __acpp_sscp_get_subgroup_max_size();
  const __acpp_uint32 wg = __acpp_sscp_get_local_size_x() * __acpp_sscp_get_local_size_y() *
                           __acpp_sscp_get_local_size_z();
  return (wg + sg - 1) / sg;
}
```

I am happy to open a PR with this if the approach is fine.
