#include <benchmark/benchmark.h>
#include "risk_engine.hpp"

// Baseline unvectorized scalar execution track to demonstrate optimization delta
int evaluate_risk_scalar(const AccountRiskMatrix* matrix, float order_qty) noexcept {
    int mask = 0;
    for (int i = 0; i < 8; ++i) {
        bool qty_pass = order_qty < matrix->max_qty_lim[i];
        bool val_pass = (order_qty * matrix->cur_asset_prices[i]) < matrix->max_val_lim[i];
        if (qty_pass && val_pass) {
            mask |= (1 << i);
        }
    }
    return mask;
}

// Global fixed matrix instance utilizing platform-agnostic intrinsic allocators
static const AccountRiskMatrix* g_matrix = []() {
    // _mm_malloc guarantees a perfectly aligned memory block on Windows/Linux/Mac
    auto* m = static_cast<AccountRiskMatrix*>(_mm_malloc(sizeof(AccountRiskMatrix), 32));
    for (int i = 0; i < 8; ++i) {
        m->max_qty_lim[i] = 500.0f;
        m->max_val_lim[i] = 10000.0f;
        m->cur_asset_prices[i] = 10.0f;
    }
    return m;
}();

static void BM_ScalarRiskCheck(benchmark::State& state) {
    for (auto _ : state) {
        int res = evaluate_risk_scalar(g_matrix, 120.0f);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_ScalarRiskCheck);

static void BM_SIMDRiskCheck(benchmark::State& state) {
    for (auto _ : state) {
        int res = evaluate_risk_simd(g_matrix, 120.0f);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_SIMDRiskCheck);

BENCHMARK_MAIN();