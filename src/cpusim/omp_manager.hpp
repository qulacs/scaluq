#pragma once
#ifdef OPENMP
#include <omp.h>

#include <core/types.hpp>

#include "omp_manager.hpp"

class OmpManager {
private:
    UINT max_num_threads_default = 1;
    UINT max_num_threads = 0;
    UINT force_threshold = 0;

    OmpManager();
    ~OmpManager() = default;

public:
    OmpManager(const OmpManager&) = delete;
    OmpManager& operator=(const OmpManager&) = delete;
    OmpManager(OmpManager&&) = delete;
    OmpManager& operator=(OmpManager&&) = delete;

    static OmpManager& get_instance() {
        static OmpManager instance;
        return instance;
    }

    void set_num_threads(int dim, UINT para_threshold);
    void reset_num_threads();
};
#endif
