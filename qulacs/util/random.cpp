#include "random.hpp"

namespace qulacs {
Random::Random() : uniform_dist(0, 1), normal_dist(0, 1) {
    std::random_device rd;
    mt.seed(rd());
}

Random::Random(UINT seed) : Random() { this->mt.seed(seed); }

double Random::uniform() { return this->uniform_dist(this->mt); }

double Random::normal() { return this->normal_dist(this->mt); }

UINT Random::int64() { return this->mt(); }

std::uint32_t Random::int32() { return this->mt() % UINT32_MAX; }
}  // namespace qulacs
