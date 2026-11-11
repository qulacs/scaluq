#include <Kokkos_Core.hpp>
#include <iostream>
int main(int argc,char**argv){ Kokkos::initialize(argc,argv); {
  auto q = Kokkos::SYCL().sycl_queue(); auto d=q.get_device();
  std::cout << "kokkos device: " << d.get_info<sycl::info::device::name>() << " local_mem=" << d.get_info<sycl::info::device::local_mem_size>() << " maxwg=" << d.get_info<sycl::info::device::max_work_group_size>() << "\n";
  std::cout << "scratch_size_max(0)=" << Kokkos::TeamPolicy<>::scratch_size_max(0) << "\n";
  Kokkos::print_configuration(std::cout, true);
 } Kokkos::finalize(); }
