#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const char* MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void empty_k(uint i [[thread_position_in_grid]]) {}

kernel void triad_k(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    y[i] = 0.5f * x[i] + y[i];
}

kernel void h_k(device float2* a [[buffer(0)]], constant uint& t [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    uint lo = i & ((1u << t) - 1u);
    uint i0 = ((i >> t) << (t + 1)) | lo;
    uint i1 = i0 | (1u << t);
    float2 u = a[i0], v = a[i1];
    const float s = 0.70710678118654752f;
    a[i0] = (u + v) * s;
    a[i1] = (u - v) * s;
}

// lo < hi are the sorted bit positions of control and target.
kernel void cnot_k(device float2* a [[buffer(0)]], constant uint4& p [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    // p.x=lo, p.y=hi, p.z=control mask, p.w=target mask
    uint j = ((i >> p.x) << (p.x + 1)) | (i & ((1u << p.x) - 1u));
    j = ((j >> p.y) << (p.y + 1)) | (j & ((1u << p.y) - 1u));
    uint i0 = j | p.z;      // control set, target 0
    uint i1 = i0 | p.w;
    float2 u = a[i0];
    a[i0] = a[i1];
    a[i1] = u;
}

#define TG 256
kernel void norm1_k(device const float2* a [[buffer(0)]], device float* partial [[buffer(1)]],
                    constant uint2& p [[buffer(2)]],  // x=N, y=num groups
                    uint tid [[thread_position_in_threadgroup]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint sid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    float s = 0.f;
    uint stride = p.y * TG;
    for (uint i = gid * TG + tid; i < p.x; i += stride) {
        float2 v = a[i];
        s += v.x * v.x + v.y * v.y;
    }
    s = simd_sum(s);
    threadgroup float sh[TG / 32];
    if (lane == 0) sh[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float r = 0.f;
        for (uint k = 0; k < TG / 32; ++k) r += sh[k];
        partial[gid] = r;
    }
}

kernel void norm2_k(device const float* partial [[buffer(0)]], device float* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint sid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    float s = 0.f;
    for (uint i = tid; i < n; i += TG) s += partial[i];
    s = simd_sum(s);
    threadgroup float sh[TG / 32];
    if (lane == 0) sh[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float r = 0.f;
        for (uint k = 0; k < TG / 32; ++k) r += sh[k];
        out[0] = r;
    }
}
)MSL";

using clk = std::chrono::steady_clock;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static MTL::Device* dev;
static MTL::CommandQueue* gq;
static MTL::ComputePipelineState *pEmpty, *pTriad, *pH, *pCnot, *pN1, *pN2;
static const uint32_t TGSZ = 256;
static const uint32_t MAX_GROUPS = 2048;

static MTL::ComputePipelineState* mkpipe(MTL::Library* lib, const char* name) {
    NS::Error* err = nullptr;
    MTL::Function* f = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!f) { fprintf(stderr, "no function %s\n", name); exit(1); }
    auto* p = dev->newComputePipelineState(f, &err);
    if (!p) { fprintf(stderr, "pipeline %s failed\n", name); exit(1); }
    f->release();
    return p;
}

static void disp(MTL::ComputeCommandEncoder* e, MTL::ComputePipelineState* p, uint64_t threads,
                 uint32_t tg = TGSZ) {
    e->setComputePipelineState(p);
    if (tg > threads) tg = (uint32_t)threads;
    e->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(tg, 1, 1));
}

static void tsv(const char* k, int n, const char* var, double ms, double gbps, const char* note) {
    if (gbps >= 0)
        printf("%s\t%d\t%s\t%.6f\t%.2f\t%s\n", k, n, var, ms, gbps, note);
    else
        printf("%s\t%d\t%s\t%.6f\t-\t%s\n", k, n, var, ms, note);
    fflush(stdout);
}

// Run a command buffer built by `enc` (which encodes `reps` ops); returns wall ms, gpu ms.
template <class F>
static void run_cb(F enc, double& wall_ms, double& gpu_ms) {
    double t0 = now_ms();
    MTL::CommandBuffer* cb = gq->commandBuffer();
    MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    enc(e);
    e->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    wall_ms = now_ms() - t0;
    gpu_ms = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
}

static void blit(MTL::Buffer* src, MTL::Buffer* dst, size_t bytes) {
    MTL::CommandBuffer* cb = gq->commandBuffer();
    MTL::BlitCommandEncoder* b = cb->blitCommandEncoder();
    b->copyFromBuffer(src, 0, dst, 0, bytes);
    b->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
}

static void fill_random(std::complex<float>* a, size_t N, unsigned seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d;
    double s = 0;
    for (size_t i = 0; i < N; i++) {
        a[i] = {d(g), d(g)};
        s += std::norm(a[i]);
    }
    float inv = (float)(1.0 / std::sqrt(s));
    for (size_t i = 0; i < N; i++) a[i] *= inv;
}

static void apply_h(MTL::ComputeCommandEncoder* e, MTL::Buffer* st, int n, uint32_t t) {
    e->setComputePipelineState(pH);
    e->setBuffer(st, 0, 0);
    e->setBytes(&t, sizeof(t), 1);
    uint64_t th = 1ull << (n - 1);
    e->dispatchThreads(MTL::Size(th, 1, 1), MTL::Size(std::min<uint64_t>(TGSZ, th), 1, 1));
}
static void apply_cnot(MTL::ComputeCommandEncoder* e, MTL::Buffer* st, int n, uint32_t c, uint32_t t) {
    uint32_t p[4] = {std::min(c, t), std::max(c, t), 1u << c, 1u << t};
    e->setComputePipelineState(pCnot);
    e->setBuffer(st, 0, 0);
    e->setBytes(p, sizeof(p), 1);
    uint64_t th = 1ull << (n - 2);
    e->dispatchThreads(MTL::Size(th, 1, 1), MTL::Size(std::min<uint64_t>(TGSZ, th), 1, 1));
}
static void apply_norm(MTL::ComputeCommandEncoder* e, MTL::Buffer* st, MTL::Buffer* part,
                       MTL::Buffer* out, int n) {
    uint64_t N = 1ull << n;
    uint32_t groups = (uint32_t)std::min<uint64_t>(MAX_GROUPS, std::max<uint64_t>(1, N / TGSZ));
    uint32_t p[2] = {(uint32_t)N, groups};
    e->setComputePipelineState(pN1);
    e->setBuffer(st, 0, 0);
    e->setBuffer(part, 0, 1);
    e->setBytes(p, sizeof(p), 2);
    e->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(TGSZ, 1, 1));
    e->setComputePipelineState(pN2);
    e->setBuffer(part, 0, 0);
    e->setBuffer(out, 0, 1);
    e->setBytes(&groups, sizeof(groups), 2);
    e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(TGSZ, 1, 1));
}

static int reps_for(int n) {
    long r = (1L << 30) >> n;  // ~ constant bytes touched
    return (int)std::max(10L, std::min(200L, r));
}

static bool check_correctness(int nc) {
    size_t N = 1ull << nc, bytes = N * 8;
    MTL::Buffer* sh = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* sh2 = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer* st = dev->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    MTL::Buffer* part = dev->newBuffer(MAX_GROUPS * 4, MTL::ResourceStorageModePrivate);
    MTL::Buffer* out = dev->newBuffer(4, MTL::ResourceStorageModeShared);
    auto* in = (std::complex<float>*)sh->contents();
    auto* res = (std::complex<float>*)sh2->contents();
    fill_random(in, N, 12345);
    bool allok = true;
    auto verify = [&](const char* name, std::vector<std::complex<double>>& ref) {
        double maxerr = 0, nrm = 0;
        for (size_t i = 0; i < N; i++) {
            maxerr = std::max(maxerr, std::abs(std::complex<double>(res[i]) - ref[i]));
            nrm += std::norm(std::complex<double>(res[i]));
        }
        // GPU norm of result
        blit(sh2, st, bytes);
        double w, g;
        run_cb([&](auto* e) { apply_norm(e, st, part, out, nc); }, w, g);
        double gn = *(float*)out->contents();
        bool ok = maxerr < 1e-6 && std::abs(nrm - 1.0) < 1e-5 && std::abs(gn - 1.0) < 1e-4;
        char note[160];
        snprintf(note, sizeof note, "maxerr=%.2e cpu_norm=%.7f gpu_norm=%.7f", maxerr, nrm, gn);
        printf("# correctness %s n=%d: %s  %s\n", name, nc, ok ? "PASS" : "FAIL", note);
        allok &= ok;
    };
    const double s2 = 1.0 / std::sqrt(2.0);
    for (uint32_t t : {0u, (uint32_t)nc / 2, (uint32_t)nc - 1}) {
        blit(sh, st, bytes);
        double w, g;
        run_cb([&](auto* e) { apply_h(e, st, nc, t); }, w, g);
        blit(st, sh2, bytes);
        std::vector<std::complex<double>> ref(N);
        for (size_t i = 0; i < N; i++) {
            size_t i0 = i & ~(1ull << t), i1 = i | (1ull << t);
            std::complex<double> u = in[i0], v = in[i1];
            ref[i] = (i & (1ull << t)) ? (u - v) * s2 : (u + v) * s2;
        }
        char nm[32];
        snprintf(nm, sizeof nm, "H(t=%u)", t);
        verify(nm, ref);
    }
    uint32_t cs[3][2] = {{(uint32_t)nc - 1, 0}, {0, (uint32_t)nc - 1}, {1, (uint32_t)nc / 2}};
    for (auto& ct : cs) {
        uint32_t c = ct[0], t = ct[1];
        blit(sh, st, bytes);
        double w, g;
        run_cb([&](auto* e) { apply_cnot(e, st, nc, c, t); }, w, g);
        blit(st, sh2, bytes);
        std::vector<std::complex<double>> ref(N);
        for (size_t i = 0; i < N; i++)
            ref[i] = (i & (1ull << c)) ? std::complex<double>(in[i ^ (1ull << t)]) : std::complex<double>(in[i]);
        char nm[32];
        snprintf(nm, sizeof nm, "CNOT(c=%u,t=%u)", c, t);
        verify(nm, ref);
    }
    sh->release(); sh2->release(); st->release(); part->release(); out->release();
    return allok;
}

int main(int argc, char** argv) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::vector<int> ns;
    for (int i = 1; i < argc; i++) ns.push_back(atoi(argv[i]));
    if (ns.empty()) ns = {16, 20, 24, 26, 28};

    dev = MTL::CreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    gq = dev->newCommandQueue();
    NS::Error* err = nullptr;
    MTL::Library* lib = dev->newLibrary(NS::String::string(MSL, NS::UTF8StringEncoding), nullptr, &err);
    if (!lib) {
        fprintf(stderr, "MSL compile failed: %s\n", err ? err->localizedDescription()->utf8String() : "?");
        return 1;
    }
    pEmpty = mkpipe(lib, "empty_k"); pTriad = mkpipe(lib, "triad_k"); pH = mkpipe(lib, "h_k");
    pCnot = mkpipe(lib, "cnot_k"); pN1 = mkpipe(lib, "norm1_k"); pN2 = mkpipe(lib, "norm2_k");
    printf("# device: %s\n", dev->name()->utf8String());
    printf("kernel\tn_qubits\tvariant\tms_per_op\tGBps\tnote\n");

    // 1. Launch latency (independent of n)
    {
        const int R = 1000;
        const uint64_t th = 256;
        for (int w = 0; w < 20; w++) {
            double a, b;
            run_cb([&](auto* e) { disp(e, pEmpty, th); }, a, b);
        }
        double t0 = now_ms(), gsum = 0;
        for (int r = 0; r < R; r++) {
            double a, b;
            run_cb([&](auto* e) { disp(e, pEmpty, th); }, a, b);
            gsum += b;
        }
        double tot = now_ms() - t0;
        tsv("launch_empty", 0, "single_cb_wall", tot / R, -1, "per cb commit+wait; R=1000");
        tsv("launch_empty", 0, "single_cb_gpu", gsum / R, -1, "GPUEnd-GPUStart avg");
        for (int w = 0; w < 3; w++) {
            double a, b;
            run_cb([&](auto* e) { for (int r = 0; r < R; r++) disp(e, pEmpty, th); }, a, b);
        }
        double wall, gpu;
        run_cb([&](auto* e) { for (int r = 0; r < R; r++) disp(e, pEmpty, th); }, wall, gpu);
        tsv("launch_empty", 0, "batched_wall", wall / R, -1, "R=1000 dispatches, one encoder, one wait");
        tsv("launch_empty", 0, "batched_gpu", gpu / R, -1, "GPUEnd-GPUStart");
    }

    bool pass = check_correctness(std::min(20, *std::max_element(ns.begin(), ns.end())));
    printf("# correctness overall: %s\n", pass ? "PASS" : "FAIL");

    for (int n : ns) {
        size_t N = 1ull << n;
        int R = reps_for(n);
        char note[128];
        double wall, gpu;
        // 2. triad
        {
            MTL::Buffer* x = dev->newBuffer(N * 4, MTL::ResourceStorageModePrivate);
            MTL::Buffer* y = dev->newBuffer(N * 4, MTL::ResourceStorageModePrivate);
            auto triad = [&](MTL::ComputeCommandEncoder* e, int reps) {
                e->setComputePipelineState(pTriad);
                e->setBuffer(x, 0, 0);
                e->setBuffer(y, 0, 1);
                for (int r = 0; r < reps; r++)
                    e->dispatchThreads(MTL::Size(N, 1, 1), MTL::Size(TGSZ, 1, 1));
            };
            run_cb([&](auto* e) { triad(e, 3); }, wall, gpu);
            run_cb([&](auto* e) { triad(e, R); }, wall, gpu);
            double bytes = 12.0 * N;
            snprintf(note, sizeof note, "R=%d y=0.5x+y float, 12B/elem", R);
            tsv("triad", n, "batched_wall", wall / R, bytes / (wall / R * 1e-3) / 1e9, note);
            tsv("triad", n, "batched_gpu", gpu / R, bytes / (gpu / R * 1e-3) / 1e9, note);
            x->release(); y->release();
        }
        MTL::Buffer* st = dev->newBuffer(N * 8, MTL::ResourceStorageModePrivate);
        MTL::Buffer* stage = dev->newBuffer(N * 8, MTL::ResourceStorageModeShared);
        MTL::Buffer* part = dev->newBuffer(MAX_GROUPS * 4, MTL::ResourceStorageModePrivate);
        MTL::Buffer* out = dev->newBuffer(4, MTL::ResourceStorageModeShared);
        auto* sp = (std::complex<float>*)stage->contents();
        fill_random(sp, N, 777);
        blit(stage, st, N * 8);
        // 3. H
        for (int t : {0, n - 1}) {
            auto f = [&](MTL::ComputeCommandEncoder* e, int reps) {
                for (int r = 0; r < reps; r++) apply_h(e, st, n, t);
            };
            run_cb([&](auto* e) { f(e, 3); }, wall, gpu);
            run_cb([&](auto* e) { f(e, R); }, wall, gpu);
            double bytes = 32.0 * (N / 2);
            snprintf(note, sizeof note, "t=%d R=%d 32B/pair", t, R);
            tsv("H", n, "batched_wall", wall / R, bytes / (wall / R * 1e-3) / 1e9, note);
            tsv("H", n, "batched_gpu", gpu / R, bytes / (gpu / R * 1e-3) / 1e9, note);
        }
        // 4. CNOT
        if (n >= 2) {
            uint32_t cts[2][2] = {{(uint32_t)n - 1, 0}, {0, (uint32_t)n - 1}};
            for (auto& ct : cts) {
                uint32_t c = ct[0], t = ct[1];
                auto f = [&](MTL::ComputeCommandEncoder* e, int reps) {
                    for (int r = 0; r < reps; r++) apply_cnot(e, st, n, c, t);
                };
                run_cb([&](auto* e) { f(e, 3); }, wall, gpu);
                run_cb([&](auto* e) { f(e, R); }, wall, gpu);
                double bytes = 32.0 * (N / 4);  // 16B read + 16B written per active pair
                snprintf(note, sizeof note, "c=%u t=%u R=%d 32B per active pair (N/4 pairs)", c, t, R);
                tsv("CNOT", n, "batched_wall", wall / R, bytes / (wall / R * 1e-3) / 1e9, note);
                tsv("CNOT", n, "batched_gpu", gpu / R, bytes / (gpu / R * 1e-3) / 1e9, note);
            }
        }
        // 5. norm reduction (state is the freshly uploaded random state)
        {
            auto f = [&](MTL::ComputeCommandEncoder* e, int reps) {
                for (int r = 0; r < reps; r++) apply_norm(e, st, part, out, n);
            };
            run_cb([&](auto* e) { f(e, 3); }, wall, gpu);
            // state was modified by H/CNOT (unitary): norm still ~1; CPU reference from original data
            double cpu = 0;
            for (size_t i = 0; i < N; i++) cpu += std::norm(std::complex<double>(sp[i]));
            run_cb([&](auto* e) { f(e, R); }, wall, gpu);
            double gv = *(float*)out->contents();
            double bytes = 8.0 * N;
            snprintf(note, sizeof note, "R=%d two-pass; rel_err_vs_cpu=%.3e (%s)", R,
                     std::abs(gv - cpu) / cpu, std::abs(gv - cpu) / cpu < 1e-4 ? "PASS" : "FAIL");
            tsv("norm", n, "batched_wall", wall / R, bytes / (wall / R * 1e-3) / 1e9, note);
            tsv("norm", n, "batched_gpu", gpu / R, bytes / (gpu / R * 1e-3) / 1e9, note);
            // single-op variant: one cb per reduction
            double t0 = now_ms();
            int S = std::min(R, 20);
            for (int r = 0; r < S; r++) {
                double a, b;
                run_cb([&](auto* e) { apply_norm(e, st, part, out, n); }, a, b);
            }
            double ms = (now_ms() - t0) / S;
            tsv("norm", n, "single_cb_wall", ms, bytes / (ms * 1e-3) / 1e9, "one cb per reduction");
        }
        st->release(); stage->release(); part->release(); out->release();
    }
    pool->release();
    return pass ? 0 : 2;
}
