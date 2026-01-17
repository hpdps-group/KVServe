"""
Test script for OnlineController.

Tests the two-tier decision making system:
1. Profile library loading
2. Analytical model predictions
3. Bandit state management
4. End-to-end controller logic
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kvserve.controller import (
    ProfileLibrary,
    OnlineController,
    AnalyticalModel,
    BanditStateManager
)


def test_profile_library():
    """Test 1: Profile library loading"""
    print("\n" + "="*60)
    print("Test 1: Profile Library Loading")
    print("="*60)
    
    library_path = "profiles/llama3.1-8b_example.json"
    library = ProfileLibrary(library_path)
    
    print(f"\n{library.summary()}")
    
    # Test bucket selection
    bucket = library.find_bucket(0.92)
    print(f"\nAccuracy 0.92 -> Bucket {bucket['bucket_id']}: {bucket['acc_range']}")
    
    # Test interval selection
    B_mbps = 150.0
    interval = library.find_interval(bucket, B_mbps)
    print(f"Bandwidth {B_mbps} MB/s -> Interval {interval['interval_id']}: {interval['B_range']} MB/s")
    
    # Test candidate retrieval
    candidates = library.get_candidate_profiles(bucket, interval)
    print(f"\nCandidates: {len(candidates)}")
    for profile in candidates:
        print(f"  - {profile}")
    
    print("\n✅ Profile library test passed!")
    return library


def test_analytical_model(library):
    """Test 2: Analytical model predictions"""
    print("\n" + "="*60)
    print("Test 2: Analytical Model")
    print("="*60)
    
    model = AnalyticalModel()
    
    # Test parameters
    V_bytes = 500 * 1024 * 1024  # 500 MB
    B_mbps = 150.0
    T_model_ms = 100.0
    
    print(f"\nTest scenario:")
    print(f"  KV volume: {V_bytes / (1024**2):.1f} MB")
    print(f"  Bandwidth: {B_mbps} MB/s")
    print(f"  Model latency: {T_model_ms} ms")
    
    # Baseline
    T_baseline = model.predict_baseline_latency(V_bytes, B_mbps, T_model_ms)
    print(f"\nBaseline latency (no compression): {T_baseline:.2f} ms")
    
    # Test all profiles
    print("\nProfile predictions:")
    for profile in library.get_all_profiles()[:5]:  # Test first 5
        T_pred = model.predict_latency(profile, V_bytes, B_mbps, T_model_ms)
        speedup = T_baseline / T_pred
        benefit = model.check_benefit(profile, B_mbps)
        
        print(f"  {profile.profile_id}:")
        print(f"    Predicted: {T_pred:.2f} ms, Speedup: {speedup:.2f}x, Benefit: {benefit}")
    
    print("\n✅ Analytical model test passed!")


def test_bandit_state():
    """Test 3: Bandit state management"""
    print("\n" + "="*60)
    print("Test 3: Bandit State Management")
    print("="*60)
    
    bandit = BanditStateManager(alpha=0.2)
    
    # Simulate observations
    bucket_id, interval_id, profile_id = 0, 1, "test_profile"
    
    print(f"\nSimulating 10 observations for ({bucket_id}, {interval_id}, {profile_id}):")
    residuals = [5.0, -2.0, 3.0, -1.0, 4.0, 2.0, -3.0, 1.0, 0.0, 2.0]
    
    for i, delta in enumerate(residuals):
        bandit.update(bucket_id, interval_id, profile_id, delta)
        N, delta_bar = bandit.get_stats(bucket_id, interval_id, profile_id)
        print(f"  Obs {i+1}: delta={delta:+.1f} ms -> N={N}, delta_bar={delta_bar:+.2f} ms")
    
    # Test save/load
    save_path = "/tmp/test_bandit_state.json"
    bandit.save(save_path)
    print(f"\n✅ State saved to {save_path}")
    
    new_bandit = BanditStateManager()
    new_bandit.load(save_path)
    N_loaded, delta_loaded = new_bandit.get_stats(bucket_id, interval_id, profile_id)
    print(f"✅ State loaded: N={N_loaded}, delta_bar={delta_loaded:+.2f} ms")
    
    print(f"\n{bandit}")
    print("\n✅ Bandit state test passed!")


def test_online_controller(library):
    """Test 4: End-to-end controller"""
    print("\n" + "="*60)
    print("Test 4: Online Controller (End-to-End)")
    print("="*60)
    
    controller = OnlineController(
        profile_library=library,
        epsilon=0.1,
        alpha=0.2
    )
    
    # Simulate 20 requests
    print("\nSimulating 20 requests:")
    print("-" * 60)
    
    for i in range(20):
        # Vary parameters
        V_bytes = (400 + i * 10) * 1024 * 1024  # 400-590 MB
        B_mbps = 100.0 + (i % 10) * 20.0         # 100-280 MB/s
        T_model_ms = 80.0 + (i % 5) * 10.0      # 80-120 ms
        T_SLO_ms = 5000.0                        # 5 seconds
        acc_req = 0.92 + (i % 3) * 0.02          # 0.92, 0.94, 0.96
        
        # Select profile
        profile, context = controller.select_profile(
            V_bytes, B_mbps, T_model_ms, T_SLO_ms, acc_req
        )
        
        if profile is None:
            print(f"Request {i+1:2d}: No compression ({context.get('reason')})")
            continue
        
        # Simulate execution with noise
        import random
        T_obs_ms = context['T_hat'] + random.gauss(0, 5.0)  # Add 5ms noise
        
        # Update bandit
        controller.update(context, T_obs_ms)
        
        # Print summary
        action = context['action']
        speedup = context.get('speedup_predicted', 1.0)
        residual = T_obs_ms - context['T_hat']
        
        print(f"Request {i+1:2d}: "
              f"B={B_mbps:5.1f} MB/s, acc={acc_req:.2f}, "
              f"profile={profile.profile_id[-2:]}, "
              f"action={action[0].upper()}, "
              f"speedup={speedup:.2f}x, "
              f"residual={residual:+.1f}ms")
    
    # Print statistics
    print("\n" + "-" * 60)
    print("Controller Statistics:")
    stats = controller.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n{controller}")
    print("\n✅ Online controller test passed!")


def test_edge_cases(library):
    """Test 5: Edge cases"""
    print("\n" + "="*60)
    print("Test 5: Edge Cases")
    print("="*60)
    
    controller = OnlineController(library, epsilon=0.1)
    
    # Case 1: Very high bandwidth (no benefit)
    print("\nCase 1: Very high bandwidth (1000 MB/s)")
    profile, context = controller.select_profile(
        V_bytes=500*1024*1024,
        B_mbps=1000.0,
        T_model_ms=100.0,
        T_SLO_ms=5000.0,
        acc_req=0.92
    )
    print(f"  Result: {profile if profile else context.get('reason')}")
    
    # Case 2: Very tight SLO
    print("\nCase 2: Very tight SLO (50 ms)")
    profile, context = controller.select_profile(
        V_bytes=500*1024*1024,
        B_mbps=150.0,
        T_model_ms=100.0,
        T_SLO_ms=50.0,
        acc_req=0.92
    )
    print(f"  Result: {profile if profile else context.get('reason')}")
    
    # Case 3: Very low bandwidth
    print("\nCase 3: Very low bandwidth (20 MB/s)")
    profile, context = controller.select_profile(
        V_bytes=500*1024*1024,
        B_mbps=20.0,
        T_model_ms=100.0,
        T_SLO_ms=50000.0,
        acc_req=0.92
    )
    print(f"  Result: {profile.profile_id if profile else context.get('reason')}")
    
    # Case 4: High accuracy requirement
    print("\nCase 4: High accuracy requirement (0.97)")
    profile, context = controller.select_profile(
        V_bytes=500*1024*1024,
        B_mbps=150.0,
        T_model_ms=100.0,
        T_SLO_ms=5000.0,
        acc_req=0.97
    )
    print(f"  Result: {profile.profile_id if profile else context.get('reason')}")
    
    print("\n✅ Edge cases test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("KVServe Online Controller Test Suite")
    print("="*60)
    
    try:
        # Test 1: Load library
        library = test_profile_library()
        
        # Test 2: Analytical model
        test_analytical_model(library)
        
        # Test 3: Bandit state
        test_bandit_state()
        
        # Test 4: End-to-end controller
        test_online_controller(library)
        
        # Test 5: Edge cases
        test_edge_cases(library)
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


