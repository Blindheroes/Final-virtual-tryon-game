"""
Main Test Runner for Virtual Try-On Game
Allows running all gesture recognition tests or individual tests
"""

import argparse
import os
import sys
import time

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_distance import run_distance_test
from test_lighting import run_lighting_test 
from test_adaptation import run_adaptation_test
from test_filtering import run_filtering_test


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition Testing Framework")
    
    # Main command selection
    parser.add_argument('--test', choices=['all', 'distance', 'lighting', 'adaptation', 'filtering'],
                       default='all', help='Test to run (default: all)')
    
    # Global parameters
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to use (default: 0)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of samples per test condition (default: 10)')
    parser.add_argument('--manual', action='store_true',
                       help='Manually record samples with the "s" key')
    parser.add_argument('--show-report', action='store_true',
                       help='Show visualization report after test completion')
    
    args = parser.parse_args()
    
    # Create test results directory if it doesn't exist
    if not os.path.exists("test_results"):
        os.makedirs("test_results")
    
    # Run selected test(s)
    if args.test == 'all' or args.test == 'distance':
        print("\n" + "="*50)
        print("STARTING DISTANCE TEST")
        print("="*50)
        run_distance_test(args)
        time.sleep(1)
    
    if args.test == 'all' or args.test == 'lighting':
        print("\n" + "="*50)
        print("STARTING LIGHTING TEST")
        print("="*50)
        run_lighting_test(args)
        time.sleep(1)
    
    if args.test == 'all' or args.test == 'adaptation':
        print("\n" + "="*50)
        print("STARTING THRESHOLD ADAPTATION TEST")
        print("="*50)
        run_adaptation_test(args)
        time.sleep(1)
    
    if args.test == 'all' or args.test == 'filtering':
        print("\n" + "="*50)
        print("STARTING TEMPORAL FILTERING TEST")
        print("="*50)
        run_filtering_test(args)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("Results saved to the test_results directory")
    print("="*50)


if __name__ == "__main__":
    main()
