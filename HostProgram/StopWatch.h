#pragma once

#include <chrono>
#include <vector>
#include <stdint.h>

using namespace std::chrono;

template <typename res>
class StopWatchTemplate {
    std::vector<typename res::time_point> m_startTPStack;
public:
    typedef enum {
        Nanoseconds,
        Microseconds,
        Milliseconds,
        Seconds,
    } EDurationType;

    typename res::time_point start() {
        typename res::time_point startTimePoint = res::now();
        m_startTPStack.push_back(startTimePoint);
        return startTimePoint;
    }

    inline uint64_t durationCast(const typename res::duration &duration, EDurationType dt) const {
        switch (dt) {
            case Nanoseconds:
                return duration_cast<nanoseconds>(duration).count();
            case Microseconds:
                return duration_cast<microseconds>(duration).count();
            case Milliseconds:
                return duration_cast<milliseconds>(duration).count();
            case Seconds:
                return duration_cast<seconds>(duration).count();
            default:
                break;
        }
        return UINT64_MAX;
    }

    uint64_t stop(EDurationType dt = EDurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.back();
        m_startTPStack.pop_back();
        return durationCast(duration, dt);
    }

    uint64_t elapsed(EDurationType dt = EDurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.back();
        return durationCast(duration, dt);
    }

    uint64_t elapsedFromRoot(EDurationType dt = EDurationType::Milliseconds) {
        typename res::duration duration = res::now() - m_startTPStack.front();
        return durationCast(duration, dt);
    }
};

typedef StopWatchTemplate<system_clock> StopWatch;
typedef StopWatchTemplate<high_resolution_clock> StopWatchHiRes;
