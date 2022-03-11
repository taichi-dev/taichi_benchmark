#include <iostream>
#include <chrono>
#include "mpm3d.cuh"

int main()
{
    using namespace std::chrono_literals;
    auto now = std::chrono::high_resolution_clock::now;
    mpm::init();
    mpm::advance();
    auto x = mpm::to_numpy();


    auto start_time = now();

    for (auto runs = 0; runs < 2048; runs++)
    {
        mpm::advance();
        auto x = mpm::to_numpy();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto used_time = (end_time - start_time) / 1ns;

    std::cout << double(used_time) / 1e9 << "s\n";
    return 0;
}
