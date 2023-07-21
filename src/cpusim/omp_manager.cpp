#ifdef OPENMP
#include "omp_manager.hpp"

#include <omp.h>

constexpr UINT MAX_NUM_THREADS = 1024;
constexpr UINT PARALLEL_NQUBIT_THRESHOLD = 64;

OmpManager::OmpManager() {
    this->max_num_threads = omp_get_max_threads();
    if (const char* tmp = std::getenv("QULACS_NUM_THREADS")) {
        const UINT tmp_val = strtol(tmp, nullptr, 0);
        if (0 < tmp_val && tmp_val <= MAX_NUM_THREADS) {
            this->max_num_threads = tmp_val;
        }
    }

    this->force_threshold = 0;
    if (const char* tmp = std::getenv("QULACS_PARALLEL_NQUBIT_THRESHOLD")) {
        const UINT tmp_val = strtol(tmp, nullptr, 0);
        if (0 < tmp_val && tmp_val <= PARALLEL_NQUBIT_THRESHOLD) {
            this->force_threshold = tmp_val;
        }
    }

    this->max_num_threads_default = omp_get_max_threads();
}

void OmpManager::set_num_threads(int dim, UINT para_threshold) {
    UINT threshold = para_threshold;
    if (this->force_threshold > 0) {
        threshold = force_threshold;
    }

    if (dim < (1 << threshold)) {
        omp_set_num_threads(1);
    } else {
        omp_set_num_threads(this->max_num_threads);
    }
}

void OmpManager::reset_num_threads() { omp_set_num_threads(this->max_num_threads_default); }
#endif
