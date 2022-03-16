#include <chrono>

class Timer {
public:
    Timer() {}
    ~Timer() {}
    
    void start() {this->startTimer = std::chrono::high_resolution_clock::now();}
    void stop() {this->stopTimer = std::chrono::high_resolution_clock::now();}
    double getTimeMillisecond() {
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stopTimer - startTimer).count();
        return duration_us / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimer;
    std::chrono::time_point<std::chrono::high_resolution_clock> stopTimer;
};
