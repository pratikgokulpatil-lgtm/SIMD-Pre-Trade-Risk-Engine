#pragma once

#include <immintrin.h>
#include <cstdint>

// Enforcing 32-byte vector bounds matching the YMM platform architecture footprint.
struct alignas(32) AccountRiskMatrix {
    float max_qty_lim[8];
    float max_val_lim[8];
    float cur_asset_prices[8];
};

static_assert(sizeof(AccountRiskMatrix) == 96, "Padding regression: Matrix size must be exactly 96 bytes.");
static_assert(alignof(AccountRiskMatrix) == 32, "Alignment regression: Matrix must sit on a strict 32-byte boundary.");

/**
 * @brief Vectorized Pre-Trade Risk Core Logic.
 * Computes limit validations across 8 accounts concurrently with zero runtime control branching.
 */
inline int evaluate_risk_simd(const AccountRiskMatrix* __restrict__ matrix, float order_qty) noexcept {
    // Broadcast the scalar parameter across all 8 parallel registers
    __m256 v_order_qty = _mm256_set1_ps(order_qty);

    // Direct hardware streaming: load memory allocations straight into execution registers
    __m256 v_max_q = _mm256_load_ps(matrix->max_qty_lim);
    __m256 v_max_v = _mm256_load_ps(matrix->max_val_lim);
    __m256 v_prices = _mm256_load_ps(matrix->cur_asset_prices);

    // Dynamic valuation matrix step: Value = Qty * Price
    __m256 v_vals = _mm256_mul_ps(v_order_qty, v_prices);

    // Compute parallel conditional bitmasks via vector comparison lanes
    __m256 qty_check = _mm256_cmp_ps(v_order_qty, v_max_q, _CMP_LT_OS);
    __m256 val_check = _mm256_cmp_ps(v_vals, v_max_v, _CMP_LT_OS);

    // Logical fusion: retain true values only if both condition parameters pass concurrently
    __m256 final_pass = _mm256_and_ps(qty_check, val_check);

    // Compress vector high bits into a compact 8-bit scalar integer representation
    return _mm256_movemask_ps(final_pass);
}