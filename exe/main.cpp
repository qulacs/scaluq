#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    scaluq::finalize();    // must be called last
}
