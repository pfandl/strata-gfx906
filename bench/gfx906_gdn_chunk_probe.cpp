// Standalone probe for gfx906's chunked GDN prompt recurrence (src/prefill/gdn_chunk_906.cu): the component time of
// gdn_recurrence() with STRATA_GDN_CHUNKED on, and how far its output sits from the shipped token recurrence.
//
// usage: gfx906_gdn_chunk_probe [T=2048] [reps=10] [candidate.bin]
//   T      tokens in the prompt chunk (the engine's is 2048)
//   reps   timed repetitions (median printed)
// The candidate output (y, y16 and the carried state) is written to the third argument for a byte-level A/B; run the
// two paths in SEPARATE processes - the dispatch caches its decisions per process.
#include "strata/prefill/kernels.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace k = strata::prefill;
namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // core/geometry.hpp's GDN shape
constexpr uint64_t kSeed = 0x67646e393036ull;
constexpr float kEps = 1e-6f;

[[noreturn]] void fail(const char* why) {
    std::fprintf(stderr, "gdn_chunk_probe: %s\n", why);
    std::exit(2);
}
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
#define HIP_CHECK(x) ck((x), #x)

struct Rng {
    uint64_t state = kSeed;
    uint32_t next() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return uint32_t((state * 2685821657736338717ull) >> 32);
    }
    float uni(float a, float b) { return a + (b - a) * (float) (next() >> 8) / 16777216.0f; }
};

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    void upload(const std::vector<T>& v) { HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice)); }
    ~Device() { if (p) (void) hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

float median_ms(std::vector<float> t) {
    std::sort(t.begin(), t.end());
    return t.size() % 2 ? t[t.size() / 2] : 0.5f * (t[t.size() / 2 - 1] + t[t.size() / 2]);
}
int64_t positive(const char* text) {
    char* end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (end == text || *end || v <= 0 || v > (1 << 20)) fail("T and reps must be positive");
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const int64_t T = argc > 1 ? positive(argv[1]) : 2048;
        const int64_t reps = argc > 2 ? positive(argv[2]) : 10;
        const char* out = argc > 3 ? argv[3] : nullptr;
        const char* env = std::getenv("STRATA_GDN_CHUNKED");
        std::printf("STRATA_GDN_CHUNKED=%s T=%lld reps=%lld\n", env ? env : "<unset>", (long long) T, (long long) reps);

        const size_t Cs = (size_t) C, Ts = (size_t) T;
        // h: q (HK*S) then k (HK*S) then v (HV*S) - the post-conv layout the recurrence reads
        std::vector<float> h(Ts * Cs);
        Rng rng;
        for (size_t t = 0; t < Ts; ++t) {
            float* row = &h[t * Cs];
            for (int i = 0; i < HK * S; ++i) row[i] = rng.uni(-1.0f, 1.0f);
            for (int i = 0; i < HK * S; ++i) row[HK * S + i] = rng.uni(-1.0f, 1.0f);
            for (int i = 0; i < HV * S; ++i) row[2 * HK * S + i] = rng.uni(-1.0f, 1.0f);
        }
        std::vector<float> gate(Ts * HV), beta(Ts * HV);
        for (size_t i = 0; i < gate.size(); ++i) {
            gate[i] = rng.uni(-3.0f, -0.05f);                              // log decay: negative, as a GDN gate
            beta[i] = 1.0f / (1.0f + std::exp(-rng.uni(-2.0f, 2.0f)));     // already sigmoided
        }
        std::vector<float> z(Ts * HV * S), wnorm(S);
        for (auto& v : z) v = rng.uni(-2.0f, 2.0f);
        for (auto& w : wnorm) w = rng.uni(0.5f, 1.5f);
        std::vector<float> state0((size_t) HV * S * S);
        for (auto& v : state0) v = rng.uni(-0.5f, 0.5f);

        Device<float> dh(h), dgate(gate), dbeta(beta), dz(z), dwnorm(wnorm), dstate(state0);
        Device<float> dy(Ts * HV * S);
        Device<uint16_t> dy16(Ts * (size_t) HV * S);
        hipStream_t st = nullptr;
        HIP_CHECK(hipStreamCreate(&st));

        std::vector<float> times;
        for (int64_t r = 0; r < reps; ++r) {
            dstate.upload(state0);   // the recurrence carries the state: every rep starts from the same one
            hipEvent_t e0, e1;
            HIP_CHECK(hipEventCreate(&e0));
            HIP_CHECK(hipEventCreate(&e1));
            HIP_CHECK(hipEventRecord(e0, st));
            k::gdn_recurrence(dstate.p, dh.p, dgate.p, dbeta.p, dz.p, dwnorm.p, kEps, dy.p, dy16.p, T, st, 0);
            HIP_CHECK(hipStreamSynchronize(st));
            HIP_CHECK(hipEventRecord(e1, st));
            HIP_CHECK(hipEventSynchronize(e1));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
            times.push_back(ms);
            HIP_CHECK(hipEventDestroy(e0));
            HIP_CHECK(hipEventDestroy(e1));
        }
        std::sort(times.begin(), times.end());
        std::printf("gdn_recurrence T=%lld: median %.3f ms  min %.3f  max %.3f\n", (long long) T,
                    (double) median_ms(times), (double) times.front(), (double) times.back());
        if (out) {
            std::vector<float> y(Ts * HV * S), stt((size_t) HV * S * S);
            std::vector<uint16_t> y16(Ts * (size_t) HV * S);
            HIP_CHECK(hipMemcpy(y.data(), dy.p, y.size() * sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(y16.data(), dy16.p, y16.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(stt.data(), dstate.p, stt.size() * sizeof(float), hipMemcpyDeviceToHost));
            FILE* f = std::fopen(out, "wb");
            if (!f) fail("cannot open the candidate file");
            std::fwrite(y.data(), sizeof(float), y.size(), f);
            std::fwrite(y16.data(), sizeof(uint16_t), y16.size(), f);
            std::fwrite(stt.data(), sizeof(float), stt.size(), f);
            std::fclose(f);
            std::printf("wrote %s (y %zu B, y16 %zu B, state %zu B)\n", out, y.size() * sizeof(float),
                        y16.size() * sizeof(uint16_t), stt.size() * sizeof(float));
        }
        HIP_CHECK(hipStreamDestroy(st));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gdn_chunk_probe: %s\n", e.what());
        return 2;
    }
}
