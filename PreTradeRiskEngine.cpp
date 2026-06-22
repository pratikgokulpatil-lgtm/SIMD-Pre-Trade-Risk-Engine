#include <bits/stdc++.h>
#include <immintrin.h>
#include <cstdint>
#include <cassert>

using namespace std;

struct alignas(32) AccountRiskMatrix{
    float max_qty_lim[8];
    float max_val_lim[8];
    float cur_asset_prices[8];
};

inline int evaluate_risk_simd(const AccountRiskMatrix* __restrict__ matrix, float order_qty){

    __m256 v_order_qty = _mm256_set1_ps(order_qty);

    __m256 v_max_q = _mm256_load_ps(matrix->max_qty_lim);
    __m256 v_max_v = _mm256_load_ps(matrix->max_val_lim);
    __m256 v_prices = _mm256_load_ps(matrix->cur_asset_prices);

    __m256 v_vals = _mm256_mul_ps(v_order_qty, v_prices);

    __m256 qty_check = _mm256_cmp_ps(v_order_qty, v_max_q, _CMP_LT_OS);
    __m256 val_check = _mm256_cmp_ps(v_vals, v_max_v, _CMP_LT_OS);

    __m256 final_pass = _mm256_and_ps(qty_check, val_check);

    return _mm256_movemask_ps(final_pass);
}

static_assert(sizeof(AccountRiskMatrix) == 96, "Padding regression: Matrix size must be 96bytes.");
static_assert(alignof(AccountRiskMatrix) == 32, "Alignment regression: Matrix must sit on a strict 32-byte boundary.");

void print_bin_status(int mask){
    for(int i=0; i<8; ++i){
        bool passed = (mask & (1<<i));
        cout << "Acc " << (i+1) << " [" << (passed ? "PASS" : "FAIL") << "]\n";
    }
    cout << "Raw Mask (Hex) : 0x" << hex << uppercase << mask << dec << "\n";
}

int main(){

    AccountRiskMatrix matrix = {
        // Max Quantity Limits per account (Acc 1 to Acc 8)
        { 500.0f,  1000.0f, 1500.0f,  100.0f,  2000.0f, 300.0f,  800.0f,  1200.0f },
        // Max Nominal Value Limits per account ($)
        { 5000.0f, 10000.0f, 15000.0f, 2000.0f, 25000.0f, 6000.0f, 18000.0f, 24000.0f },
        // Current Asset Prices seen by each account ($ per unit)
        { 10.0f,   10.0f,    10.0f,    50.0f,   10.0f,    20.0f,   15.0f,    12.0f   }
    };

    float qty = 120.0f;

    int res = evaluate_risk_simd(&matrix, qty);
    print_bin_status(res);
    return 0;
}