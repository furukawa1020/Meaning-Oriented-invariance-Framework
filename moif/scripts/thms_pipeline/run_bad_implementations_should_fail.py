import traceback
import sys
from test_leakage_adversarial import TestAdversarialLeakage

def run_failing_tests():
    print("--- BAD IMPLEMENTATION FAILURE LOG ---")
    print("This script proves that the adversarial tests successfully catch leaky implementations.\n")
    
    test_instance = TestAdversarialLeakage()
    test_instance.setUpClass()
    
    bad_tests = [
        ("test_01_active_train_perturbation_fails_bad_baseline", test_instance.test_01_active_train_perturbation_fails_bad_baseline),
        ("test_03_test_perturbation_fails_bad_baseline", test_instance.test_03_test_perturbation_fails_bad_baseline),
        ("test_05_future_spike_fails_centered_rolling", test_instance.test_05_future_spike_fails_centered_rolling),
        ("test_07_heldout_subject_fails_bad_population", test_instance.test_07_heldout_subject_fails_bad_population)
    ]
    
    passed_when_should_fail = 0
    caught_leaks = 0
    
    for name, test_func in bad_tests:
        print(f"Running {name}...")
        try:
            test_func()
            print(f"[SUCCESS] Test passed! The leakage was CAUGHT by the assertions.")
            caught_leaks += 1
        except AssertionError as e:
            print(f"[ERROR] Assertion failed: {e}. The leakage bypassed the test!")
            passed_when_should_fail += 1
        except Exception as e:
            print(f"[ERROR] Test crashed for an unexpected reason: {e}")
            passed_when_should_fail += 1
            
        print("-" * 40)
        
    print("\n--- SUMMARY ---")
    print(f"Intentionally Leaky Implementations Caught: {caught_leaks}/{len(bad_tests)}")
    if passed_when_should_fail > 0:
        print("RED TEAM ALERT: Some leaky implementations bypassed the tests.")
        sys.exit(1)
    else:
        print("Verification Passed: The adversarial test suite has teeth.")

if __name__ == "__main__":
    run_failing_tests()
