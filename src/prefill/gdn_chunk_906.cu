// src/prefill/gdn_chunk_906.cu - the chunked GDN prompt recurrence for gfx906 (design: docs/GDN_CHUNK_SCAN.md).
//
// The same WY form as src/prefill/kernels.cu's gdn_rec_chunked (which is CUDA-only: the pair sits inside
// `#if !defined(__HIPCC__)`, the launcher demands sm_80+, 128 SMs and 93.7 KB of opt-in LDS, and the prep's
// `if (warp >= VPK) return;` + `__syncwarp()` is illegal on a wave64).  This is that pair reshaped for a card with
// 64 KB of LDS, no cp.async and 64-lane waves:
//
//   gdn906_prep_kernel  per (chunk, key head): gamma per token, the decayed Gram matrices k_t.k_i / q_t.k_i, the
//                       unit-lower-triangular inverse T = (I + A)^-1 and P, for the key head's three value heads
//   gdn906_scan_kernel  per (key head, 8 value columns): walks the super-block's chunks in order, the state slice in
//                       registers, every per-token product a small dense matmul over the staged k and q tiles
//
// Differences from the CUDA pair, all forced by the part: the scan takes GDV = 8 value columns per block (GDV = 16
// with a double-buffered k needs 91.5 KB of LDS), k is single-buffered (staged again after the state update instead of
// while the chunk computes), the solve is sequenced with __syncthreads() instead of __syncwarp() and no 32-lane group
// ever leaves the block, and every 32-lane shuffle carries width 32 (the solve's three groups are not all in the lower
// half of a wave, so HIP's default wave-wide width would broadcast the wrong lane).
//
// Same numbers up to FP32 rounding (another order of the same sums), not bit-exact: docs/GDN_CHUNK_SCAN.md.
#include "strata/prefill/gdn_chunk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill {
#if defined(__HIPCC__)
namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // core/geometry.hpp
constexpr int VPK = HV / HK;                          // 3 value heads per key head
constexpr int GCH = 32;                               // tokens per chunk (a lane per token in the solve)
constexpr int GDV = 8;                                // value columns per scan block
constexpr int GSB = 1024;                             // tokens per super-block
constexpr int PSP = S + 1;                            // padded k / q rows in the prep
constexpr int GTP = GCH + 1;                          // padded T / P rows in the prep
constexpr int GKP = S + 4;                            // padded k / q rows in the scan (8 rows of a wave on 8 banks)
constexpr int NT = 128 * VPK;                         // scan threads: a 128-thread group per value head
static_assert(GCH == 32, "gdn906_prep_kernel: a lane per token of the chunk");
static_assert(S % GDV == 0, "gdn906_scan_kernel: the column blocks must tile S");
static_assert(VPK * GCH * GTP <= 2 * GCH * PSP, "gdn906_prep_kernel: A has to fit where k and q were");

constexpr size_t kPrepSmem = sizeof(float) * (2 * GCH * PSP + 2 * GCH * GTP);
constexpr size_t kScanSmem = sizeof(float) * (2 * GCH * GKP + VPK * (S * GDV + 3 * GCH * GDV));
static_assert(kScanSmem <= 64 * 1024, "gfx906: 64 KB of LDS per block");
static_assert(kPrepSmem <= 64 * 1024, "gfx906: 64 KB of LDS per block");

[[noreturn]] void gdn906_fail(const char* what) {
    std::fprintf(stderr, "gdn_chunk_906 %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
    std::exit(1);
}

// Per (chunk, key head): gamma (the gate cumsum inside the chunk), A/P (the decayed Grams), T = (I + A)^-1 and P
// decayed - the three matrices the scan's solve and output sum read.  256 threads: the k/q stage and the products use
// all eight 32-lane groups; the solve uses three of them (one per value head of the key head) and the other five wait
// at the same barriers, which is what keeps the wave64 case legal.
__global__ void __launch_bounds__(256) gdn906_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                         const float* __restrict__ beta, float* __restrict__ tm,
                                                         float* __restrict__ pm, float* __restrict__ gm, int64_t t0,
                                                         int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    float* sk = gsm;                       // [GCH][PSP]
    float* sq = sk + GCH * PSP;            // [GCH][PSP]
    float* skk = sq + GCH * PSP;           // [GCH][GTP]  k_t.k_i
    float* sqk = skk + GCH * GTP;          // [GCH][GTP]  q_t.k_i
    float* sa = gsm;                       // [VPK][GCH][GTP]  A, then T below the diagonal (over k and q)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, hw = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * GCH;
    const int n = (int) min((int64_t) GCH, T - tb);
    {   // k and q as float4, every load issued before the first store
        constexpr int NP = GCH * S / 4 / 256;   // float4 a thread, each of k and q
        float4 kv[NP], qv[NP];
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            kv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + HK * S + qh * S + r)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = t < n ? *reinterpret_cast<const float4*>(h + (tb + t) * C + qh * S + r)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m, t = p / (S / 4), r = (p % (S / 4)) * 4;
            float* dk = sk + t * PSP + r;
            float* dq = sq + t * PSP + r;
            dk[0] = kv[m].x; dk[1] = kv[m].y; dk[2] = kv[m].z; dk[3] = kv[m].w;
            dq[0] = qv[m].x; dq[1] = qv[m].y; dq[2] = qv[m].z; dq[3] = qv[m].w;
        }
    }
    __syncthreads();
    {   // lane i, rows hw + 8m: k_t.k_i and q_t.k_i on eight independent sums (rows 129 apart: 32 banks)
        constexpr int RW = GCH / 8;
        float kk[RW] = {}, qk[RW] = {};
        for (int r = 0; r < S; ++r) {
            const float ki = sk[lane * PSP + r];
#pragma unroll
            for (int m = 0; m < RW; ++m) {
                kk[m] = fmaf(sk[(hw + 8 * m) * PSP + r], ki, kk[m]);
                qk[m] = fmaf(sq[(hw + 8 * m) * PSP + r], ki, qk[m]);
            }
        }
#pragma unroll
        for (int m = 0; m < RW; ++m) {
            const int t = hw + 8 * m;
            skk[t * GTP + lane] = lane <= t ? kk[m] : 0.0f;
            sqk[t * GTP + lane] = lane <= t ? qk[m] : 0.0f;
        }
    }
    __syncthreads();
    // the solve: VPK of the 8 groups, one value head each; every group meets every barrier below
    const bool solve = hw < VPK;
    const int vh = qh + hw * HK;
    float* a = sa + (solve ? hw : 0) * GCH * GTP;
    float b = 0.0f, gam = 0.0f;
    if (solve) {
        b = lane < n ? beta[(tb + lane) * HV + vh] : 0.0f;
        gam = lane < n ? gate[(tb + lane) * HV + vh] : 0.0f;   // past the end: gate 0, beta 0 (no effect on S)
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o, 32);
        if (lane >= o) gam += u;
    }
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t, 32), bt = __shfl_sync(0xffffffffu, b, t, 32);
        if (solve)
            a[t * GTP + lane] = lane < t ? bt * __expf(gt - gam) * skk[t * GTP + lane] : (lane == t ? 1.0f : 0.0f);
    }
    __syncthreads();
    // T = (I + A)^-1 by forward substitution, row by row in place: T[t][i] = -sum_{k < t} A[t][k] T[k][i] over the
    // finished rows (diagonal 1, zeros above it: the terms k < i add an exact 0).  A trip the whole block shares -
    // __syncthreads(), because __syncwarp() in a block that five groups have left is not a thing on a wave64.
    for (int t = 1; t < GCH; ++t) {
        float v = 0.0f;
        if (solve) {
#pragma unroll 8
            for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTP + k], a[k * GTP + lane], v);
        }
        __syncthreads();
        if (solve && lane < t) a[t * GTP + lane] = v;
        __syncthreads();
    }
    if (!solve) return;   // the last barrier of the block has been passed
    float* T_ = tm + ((int64_t) ch * HV + vh) * GCH * GCH;
    float* P_ = pm + ((int64_t) ch * HV + vh) * GCH * GCH;
    for (int t = 0; t < GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t, 32);
        T_[t * GCH + lane] = a[t * GTP + lane];
        P_[t * GCH + lane] = lane <= t ? __expf(gt - gam) * sqk[t * GTP + lane] : 0.0f;
    }
    gm[((int64_t) ch * HV + vh) * GCH + lane] = gam;
}

// The chunk-serial scan: one block per (key head, GDV value columns), a 128-thread group per value head of that key
// head, the state slice in registers across the super-block's chunks.  Per chunk: X = K S0 and Q S0 for this thread's
// token (a 4-row x 2-column tile of every product), Y = beta (V - exp(gamma) X), D = T Y, o = exp(gamma) Q S0 + P D,
// then S = exp(gamma_last) S0 + K^T (D exp(gamma_last - gamma)).  Four barriers a chunk, and the next chunk's k and q
// are staged after the update that reads them (single buffer: the double buffer of the CUDA kernel is 17 KB this card
// does not have to spare).
__global__ void __launch_bounds__(NT) gdn906_scan_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                        const float* __restrict__ beta, const float* __restrict__ tm,
                                                        const float* __restrict__ pm, const float* __restrict__ gm,
                                                        float* __restrict__ oc, int64_t t0, int nch, int64_t T) {
    extern __shared__ __align__(16) float gsm[];
    constexpr int HEAD = S * GDV + 3 * GCH * GDV;   // a value head's shared floats
    float* kbuf = gsm;                              // [GCH][GKP]
    float* sq = kbuf + GCH * GKP;                   // [GCH][GKP]
    float* shd = sq + GCH * GKP;                    // [VPK][HEAD]  state slice, Y, D, decayed D
    const int qh = blockIdx.x, c0 = blockIdx.y * GDV;
    const int tid = threadIdx.x, gi = tid >> 7, t128 = tid & 127;
    const int vh = qh + gi * HK;
    const int tt = t128 >> 2, vv = 2 * (t128 & 3), rq = tt;   // this token; its 2 columns; its row group
    float* ss = shd + gi * HEAD;                             // [S][GDV]  the state slice at the chunk's start
    float* sy = ss + S * GDV;                                // [GCH][GDV]  Y
    float* sd = sy + GCH * GDV;                              // [GCH][GDV]  D
    float* sdd = sd + GCH * GDV;                             // [GCH][GDV]  exp(gamma_last - gamma) D
    const float* kbase = h + HK * S + qh * S;
    const float* qbase = h + qh * S;
    const float rsqrt_S = rsqrtf((float) S);
    auto stage_kq = [&](int64_t c_t0, int n) {   // k and q rows of a chunk, every load before its store
        for (int p = tid; p < GCH * (S / 4); p += NT) {
            const int t = p / (S / 4), r = (p % (S / 4)) * 4;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), qv = kv;
            if (t < n) {
                kv = *reinterpret_cast<const float4*>(kbase + (c_t0 + t) * C + r);
                qv = *reinterpret_cast<const float4*>(qbase + (c_t0 + t) * C + r);
            }
            *reinterpret_cast<float4*>(&kbuf[t * GKP + r]) = kv;
            *reinterpret_cast<float4*>(&sq[t * GKP + r]) = qv;
        }
    };
    float s[4][2];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
#pragma unroll
        for (int y = 0; y < 2; ++y) s[x][y] = state[((size_t) (4 * rq + x) * HV + vh) * S + c0 + vv + y];
        *reinterpret_cast<float2*>(&ss[(4 * rq + x) * GDV + vv]) = make_float2(s[x][0], s[x][1]);
    }
    stage_kq(t0, (int) min((int64_t) GCH, T - t0));
    __syncthreads();
    for (int ch = 0; ch < nch; ++ch) {
        const int64_t c_t0 = t0 + (int64_t) ch * GCH;
        const int n = (int) min((int64_t) GCH, T - c_t0);
        // this token's v, gamma, beta and the chunk's last gamma
        float v[2];
        v[0] = tt < n ? h[(c_t0 + tt) * C + 2 * HK * S + vh * S + c0 + vv] : 0.0f;
        v[1] = tt < n ? h[(c_t0 + tt) * C + 2 * HK * S + vh * S + c0 + vv + 1] : 0.0f;
        const float* g = gm + ((int64_t) ch * HV + vh) * GCH;
        const float gam = g[tt], glast = g[GCH - 1];
        const float bet = tt < n ? beta[(c_t0 + tt) * HV + vh] : 0.0f;
        // this token's T and P rows: read from L2 inside the two sums below rather than held in 64 registers - a chunk's
        // T and P are 8 KB that the 16 column blocks of a head all read, and 64 live floats per thread cost the
        // occupancy the scan needs more than the reads cost it (measured: the register version spilled, and the phase
        // ran 4.36 ms against 5.2 ms for the token recurrence it has to beat)
        const float* Trow = tm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH;
        const float* Prow = pm + (((int64_t) ch * HV + vh) * GCH + tt) * GCH;
        // X = K S0 and Q S0 for this token, two columns
        float x[2] = {0.0f, 0.0f}, qs2[2] = {0.0f, 0.0f};
#pragma unroll 4
        for (int r = 0; r < S; r += 4) {
            const float4 k4 = *reinterpret_cast<const float4*>(&kbuf[tt * GKP + r]);
            const float4 q4 = *reinterpret_cast<const float4*>(&sq[tt * GKP + r]);
            const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const float2 s2 = *reinterpret_cast<const float2*>(&ss[(r + jj) * GDV + vv]);
                x[0] = fmaf(kk[jj], s2.x, x[0]);
                x[1] = fmaf(kk[jj], s2.y, x[1]);
                qs2[0] = fmaf(qq[jj], s2.x, qs2[0]);
                qs2[1] = fmaf(qq[jj], s2.y, qs2[1]);
            }
        }
        const float e = __expf(gam);
        *reinterpret_cast<float2*>(&sy[tt * GDV + vv]) = make_float2(bet * (v[0] - e * x[0]), bet * (v[1] - e * x[1]));
        __syncthreads();   // (A) Y complete; every read of q done
        // D = T Y (T is zero above its diagonal), and D decayed to the chunk's end for the update
        float d[2] = {0.0f, 0.0f};
#pragma unroll 8
        for (int i = 0; i < GCH; ++i) {
            const float ti = __ldg(Trow + i);
            const float2 y2 = *reinterpret_cast<const float2*>(&sy[i * GDV + vv]);
            d[0] = fmaf(ti, y2.x, d[0]);
            d[1] = fmaf(ti, y2.y, d[1]);
        }
        const float fdec = __expf(glast - gam);
        *reinterpret_cast<float2*>(&sd[tt * GDV + vv]) = make_float2(d[0], d[1]);
        *reinterpret_cast<float2*>(&sdd[tt * GDV + vv]) = make_float2(fdec * d[0], fdec * d[1]);
        __syncthreads();   // (B) D complete
        // o = exp(gamma) Q S0 + P D (P is zero above its diagonal)
        float o[2] = {e * qs2[0], e * qs2[1]};
#pragma unroll 8
        for (int i = 0; i < GCH; ++i) {
            const float pi = __ldg(Prow + i);
            const float2 d2 = *reinterpret_cast<const float2*>(&sd[i * GDV + vv]);
            o[0] = fmaf(pi, d2.x, o[0]);
            o[1] = fmaf(pi, d2.y, o[1]);
        }
        if (tt < n) {
            float* row = oc + (c_t0 + tt) * HV * S + vh * S + c0 + vv;
            row[0] = o[0] * rsqrt_S;
            row[1] = o[1] * rsqrt_S;
        }
        // S = exp(gamma_last) S0 + K^T (decayed D)
        const float gL = __expf(glast);
#pragma unroll
        for (int xx = 0; xx < 4; ++xx) { s[xx][0] *= gL; s[xx][1] *= gL; }
#pragma unroll 4
        for (int t = 0; t < GCH; ++t) {
            const float4 k4 = *reinterpret_cast<const float4*>(&kbuf[t * GKP + 4 * rq]);
            const float2 d2 = *reinterpret_cast<const float2*>(&sdd[t * GDV + vv]);
            const float kx[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
            for (int xx = 0; xx < 4; ++xx) {
                s[xx][0] = fmaf(kx[xx], d2.x, s[xx][0]);
                s[xx][1] = fmaf(kx[xx], d2.y, s[xx][1]);
            }
        }
#pragma unroll
        for (int xx = 0; xx < 4; ++xx)
            *reinterpret_cast<float2*>(&ss[(4 * rq + xx) * GDV + vv]) = make_float2(s[xx][0], s[xx][1]);
        __syncthreads();   // (C) the state slice is in LDS and every read of k is done
        if (ch + 1 < nch) {
            const int64_t n_t0 = c_t0 + GCH;
            stage_kq(n_t0, (int) min((int64_t) GCH, T - n_t0));
            __syncthreads();   // (D) the next chunk's k and q landed
        }
    }
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 2; ++y) state[((size_t) (4 * rq + x) * HV + vh) * S + c0 + vv + y] = s[x][y];
}

// The scratch: T, P and gamma of one super-block's chunks, per device, allocated once at its largest and kept (a
// stream-ordered allocation and free per call costs host time the GPU waits for - the CUDA path measured 78 ms against
// 34 ms for exactly that).
float* gdn906_scratch() {
    static float* bufs[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return nullptr;
    }
    if (bufs[dev] == nullptr) {
        const size_t n = (size_t) (GSB / GCH) * HV * (2 * GCH * GCH + GCH);
        if (cudaMalloc((void**) &bufs[dev], n * sizeof(float)) != cudaSuccess) {
            (void) cudaGetLastError();
            bufs[dev] = nullptr;
        }
    }
    return bufs[dev];
}

// gfx906 only: this kernel shape is sized for 64 KB of LDS and its tile maps were chosen for a 60-CU part.  Cached per
// device (a layer split runs the recurrence on more than one card).
bool gdn906_arch_ok() {
    static int ok[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return false;
    }
    if (ok[dev] == 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        const bool is906 = std::strncmp(prop.gcnArchName, "gfx906", 6) == 0;
        ok[dev] = is906 ? 1 : -1;
        if (is906) {
            // the scan's 55 KB are past the 48 KB default: opt in once per device, and treat a refusal as "not this card"
            if (cudaFuncSetAttribute(gdn906_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kScanSmem) !=
                cudaSuccess) {
                (void) cudaGetLastError();
                ok[dev] = -1;
            }
        }
    }
    return ok[dev] > 0;
}

}  // namespace
#endif  // __HIPCC__

bool gdn_chunk_gfx906(float* state, const float* h, const float* gate, const float* beta, float* oc, int64_t T,
                      void* stream) {
#if defined(__HIPCC__)
    if (T <= 0 || state == nullptr || h == nullptr || gate == nullptr || beta == nullptr || oc == nullptr) return false;
    if (!gdn906_arch_ok()) return false;
    float* scratch = gdn906_scratch();
    if (scratch == nullptr) return false;
    const int nch_sb = GSB / GCH;
    const size_t tsz = (size_t) nch_sb * HV * GCH * GCH;
    float* tm = scratch, *pm = tm + tsz, *gm = pm + tsz;
    cudaStream_t cs = (cudaStream_t) stream;
    // STRATA_GDN_CHUNK_PROF=1: per-stage milliseconds over the whole call (a dev tool, not a hot path - the same
    // switch the Volta path has, and it is what says whether the prep or the scan is the phase)
    static const bool prof = std::getenv("STRATA_GDN_CHUNK_PROF") != nullptr;
    cudaEvent_t pe0{}, pe1{};
    float totp = 0.0f, tots = 0.0f;
    if (prof) {
        cudaEventCreate(&pe0);
        cudaEventCreate(&pe1);
    }
    for (int64_t t0 = 0; t0 < T; t0 += GSB) {
        const int nch = (int) ((std::min<int64_t>(GSB, T - t0) + GCH - 1) / GCH);
        if (prof) cudaEventRecord(pe0, cs);
        gdn906_prep_kernel<<<dim3((unsigned) nch, HK), 256, kPrepSmem, cs>>>(h, gate, beta, tm, pm, gm, t0, T);
        if (cudaGetLastError() != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (prof) {
            cudaEventRecord(pe1, cs);
            cudaEventSynchronize(pe1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, pe0, pe1);
            totp += ms;
            cudaEventRecord(pe0, cs);
        }
        gdn906_scan_kernel<<<dim3(HK, S / GDV), NT, kScanSmem, cs>>>(state, h, beta, tm, pm, gm, oc, t0, nch, T);
        if (cudaGetLastError() != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (prof) {
            cudaEventRecord(pe1, cs);
            cudaEventSynchronize(pe1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, pe0, pe1);
            tots += ms;
        }
    }
    if (prof) {
        std::printf("gdn_chunk_906 prof T=%lld: prep %.3fms scan %.3fms\n", (long long) T, (double) totp, (double) tots);
        cudaEventDestroy(pe0);
        cudaEventDestroy(pe1);
    }
    return true;
#else
    (void) state; (void) h; (void) gate; (void) beta; (void) oc; (void) T; (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill
