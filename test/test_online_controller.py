#!/usr/bin/env python3
"""
Test script for online controller with dynamic profile loading.

Measures the overhead of online profile selection including:
- Profile loading and building
- Throughput lookup
- Candidate filtering
- Latency prediction
- Final selection
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kvserve.controller.dynamic_online_controller import DynamicOnlineController


def test_single_selection():
    """Test a single profile selection and measure overhead."""
    print("=" * 80)
    print("Test 1: Single Profile Selection")
    print("=" * 80)
    
    # Initialize controller
    print("\n[1] Initializing controller...")
    init_start = time.perf_counter()
    
    controller = DynamicOnlineController(
        model_name="Qwen2.5-7B-Instruct",
        dataset="qasper",
        machine="5090",
        epsilon=0.0,  # Pure exploitation (no exploration)
        max_candidates=3
    )
    
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"   Initialization time: {init_time:.3f} ms")
    
    # Test parameters
    input_length = 4096  # tokens
    V_bytes = 100 * 1024 * 1024  # 100 MB KV cache
    B_mbps = 1000.0  # 1 GB/s network bandwidth (= 1000 MB/s)
    T_model_ms = 50.0  # 50 ms model computation
    acc_req = 95.0  # 95% accuracy requirement
    
    print(f"\n[2] Test parameters:")
    print(f"   Input length: {input_length} tokens")
    print(f"   KV cache size: {V_bytes / (1024*1024):.1f} MB")
    print(f"   Bandwidth: {B_mbps} MB/s")
    print(f"   Model latency: {T_model_ms} ms")
    print(f"   Accuracy requirement: {acc_req}%")
    
    # Select profile
    print(f"\n[3] Selecting profile...")
    select_start = time.perf_counter()
    
    profile, context = controller.select_profile(
        input_length=input_length,
        V_bytes=V_bytes,
        B_mbps=B_mbps,
        T_model_ms=T_model_ms,
        acc_req=acc_req
    )
    
    select_time = (time.perf_counter() - select_start) * 1000
    
    # Display results
    print(f"\n[4] Selection results:")
    if profile is None:
        print(f"   Selected: No compression")
        print(f"   Reason: {context.get('reason', 'unknown')}")
    else:
        print(f"   Selected profile: {profile.profile_id}")
        print(f"   Compression ratio: {profile.compression_ratio:.2f}x")
        print(f"   Accuracy: {profile.accuracy:.2f}%")
        print(f"   Harmonic speed: {profile.harmonic_speed:.1f} MB/s")
        print(f"   Critical bandwidth: {profile.critical_bandwidth:.1f} MB/s")
        print(f"   Predicted latency: {context['T_hat']:.2f} ms")
        print(f"   Baseline latency: {context['baseline_latency']:.2f} ms")
        print(f"   Predicted speedup: {context['speedup_predicted']:.2f}x")
        print(f"   Num candidates: {context['num_candidates']}")
    
    print(f"\n[5] Performance metrics:")
    print(f"   Selection time: {select_time:.3f} ms")
    print(f"   Controller internal time: {context.get('selection_time_ms', 0):.3f} ms")
    
    return select_time


def test_multiple_selections():
    """Test multiple selections with different input lengths."""
    print("\n" + "=" * 80)
    print("Test 2: Multiple Selections (Different Input Lengths)")
    print("=" * 80)
    
    controller = DynamicOnlineController(
        model_name="Qwen2.5-7B-Instruct",
        dataset="qasper",
        machine="5090",
        epsilon=0.0,
        max_candidates=3
    )
    
    # Test with different input lengths
    input_lengths = [1024, 2048, 4096, 8192, 16384]
    V_bytes = 100 * 1024 * 1024
    B_mbps = 1000.0
    T_model_ms = 50.0
    acc_req = 95.0
    
    print(f"\n[1] Running {len(input_lengths)} selections...")
    
    for i, input_length in enumerate(input_lengths):
        profile, context = controller.select_profile(
            input_length=input_length,
            V_bytes=V_bytes,
            B_mbps=B_mbps,
            T_model_ms=T_model_ms,
            acc_req=acc_req
        )
        
        if profile:
            print(f"   [{i+1}] Length={input_length:5d}: "
                  f"CR={profile.compression_ratio:.2f}x, "
                  f"Speed={profile.harmonic_speed:6.1f} MB/s, "
                  f"Time={context['selection_time_ms']:.3f} ms")
        else:
            print(f"   [{i+1}] Length={input_length:5d}: No compression, "
                  f"Time={context['selection_time_ms']:.3f} ms")
    
    # Show statistics
    stats = controller.get_statistics()
    print(f"\n[2] Summary statistics:")
    print(f"   Total selections: {stats['total_decisions']}")
    print(f"   Average time: {stats['avg_selection_time_ms']:.3f} ms")
    print(f"   Min time: {stats['min_selection_time_ms']:.3f} ms")
    print(f"   Max time: {stats['max_selection_time_ms']:.3f} ms")
    print(f"   Cached lengths: {stats['cached_input_lengths']}")


def test_warm_vs_cold():
    """Test cold start vs warm cache performance."""
    print("\n" + "=" * 80)
    print("Test 3: Cold Start vs Warm Cache")
    print("=" * 80)
    
    controller = DynamicOnlineController(
        model_name="Qwen2.5-7B-Instruct",
        dataset="qasper",
        machine="5090"
    )
    
    input_length = 4096
    V_bytes = 100 * 1024 * 1024
    B_mbps = 1000.0
    T_model_ms = 50.0
    acc_req = 95.0
    
    # Cold start
    print("\n[1] Cold start (first selection)...")
    cold_start = time.perf_counter()
    
    profile1, context1 = controller.select_profile(
        input_length, V_bytes, B_mbps, T_model_ms, acc_req
    )
    
    cold_time = (time.perf_counter() - cold_start) * 1000
    print(f"   Cold start time: {cold_time:.3f} ms")
    
    # Warm cache (same input length)
    print("\n[2] Warm cache (same input length)...")
    warm_times = []
    
    for i in range(10):
        warm_start = time.perf_counter()
        
        profile2, context2 = controller.select_profile(
            input_length, V_bytes, B_mbps, T_model_ms, acc_req
        )
        
        warm_time = (time.perf_counter() - warm_start) * 1000
        warm_times.append(warm_time)
    
    avg_warm_time = sum(warm_times) / len(warm_times)
    print(f"   Average warm time: {avg_warm_time:.3f} ms (over {len(warm_times)} runs)")
    print(f"   Speedup: {cold_time / avg_warm_time:.2f}x")


def test_different_scenarios():
    """Test different bandwidth and accuracy scenarios."""
    print("\n" + "=" * 80)
    print("Test 4: Different Scenarios")
    print("=" * 80)
    
    controller = DynamicOnlineController(
        model_name="Qwen2.5-7B-Instruct",
        dataset="qasper",
        machine="5090"
    )
    
    input_length = 4096
    V_bytes = 100 * 1024 * 1024
    T_model_ms = 50.0
    
    scenarios = [
        {"B_mbps": 500, "acc_req": 95.0, "name": "Low BW, High Acc"},
        {"B_mbps": 2000, "acc_req": 95.0, "name": "High BW, High Acc"},
        {"B_mbps": 500, "acc_req": 90.0, "name": "Low BW, Low Acc"},
        {"B_mbps": 2000, "acc_req": 90.0, "name": "High BW, Low Acc"},
    ]
    
    print(f"\n[1] Testing {len(scenarios)} scenarios...")
    print()
    
    for i, scenario in enumerate(scenarios):
    profile, context = controller.select_profile(
            input_length=input_length,
            V_bytes=V_bytes,
            B_mbps=scenario["B_mbps"],
            T_model_ms=T_model_ms,
            acc_req=scenario["acc_req"]
        )
        
        print(f"[{i+1}] {scenario['name']}:")
        print(f"    B={scenario['B_mbps']} MB/s, Acc≥{scenario['acc_req']}%")
        
        if profile:
            print(f"    → Profile: CR={profile.compression_ratio:.2f}x, "
                  f"Acc={profile.accuracy:.2f}%, "
                  f"Speedup={context['speedup_predicted']:.2f}x")
        else:
            print(f"    → No compression ({context.get('reason', 'unknown')})")
    
        print(f"    Selection time: {context['selection_time_ms']:.3f} ms")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ONLINE CONTROLLER PERFORMANCE TEST")
    print("=" * 80)
    
    try:
        # Test 1: Single selection
        test_single_selection()
        
        # Test 2: Multiple selections
        test_multiple_selections()
        
        # Test 3: Cold vs warm
        test_warm_vs_cold()
        
        # Test 4: Different scenarios
        test_different_scenarios()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
