#include <iostream>
#include <iomanip>
#include "risk_engine.hpp"

void print_bin_status(int mask) noexcept {
    std::cout << "Engine Evaluation Matrix Breakdown:\n";
    for (int i = 0; i < 8; ++i) {
        bool passed = (mask & (1 << i));
        std::cout << "  Account " << (i + 1) << ": [" << (passed ? "PASS" : "FAIL") << "]\n";
    }
    std::cout << "Aggregated Output Bitmap Mask: 0x" 
              << std::hex << std::uppercase << mask << std::dec << "\n\n";
}

int main() {
    std::cout << "Running Low-Latency Engine Boundary Tests...\n\n";

    // Setup an explicit test environment profile
    // Account 4 (Index 3) represents an intensely risk-constrained venue profile.
    AccountRiskMatrix matrix = {
        { 500.0f,  1000.0f, 1500.0f,  100.0f,  2000.0f, 300.0f,  800.0f,  1200.0f },
        { 5000.0f, 10000.0f, 15000.0f, 2000.0f, 25000.0f, 6000.0f, 18000.0f, 24000.0f },
        { 10.0f,   10.0f,    10.0f,    50.0f,   10.0f,    20.0f,   15.0f,    12.0f   }
    };

    // Test Scenario: Inbound liquidity size crosses Account 4's custom limits
    float order_qty = 120.0f;
    std::cout << "Inbound Execution Vector Sample Size: " << order_qty << " units.\n";

    int res = evaluate_risk_simd(&matrix, order_qty);
    print_bin_status(res);

    return 0;
}