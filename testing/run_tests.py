"""
Main testing module that can run all tests or individual tests based on command-line arguments.
"""

from test_gesture_distance import test_gesture_distance
from test_gesture_lighting import test_gesture_lighting
from test_adaptive_threshold import test_adaptive_threshold
from test_temporal_filtering import test_temporal_filtering
import argparse
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules after adding parent directory to path


def main():
    parser = argparse.ArgumentParser(
        description='Virtual Try-On Game Testing Suite')
    parser.add_argument('--test', choices=['distance', 'lighting', 'threshold', 'temporal', 'all'],
                        default='all', help='Which test to run')

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    if args.test == 'distance' or args.test == 'all':
        print("\n=== Running Hand Gesture Distance Test ===")
        test_gesture_distance()

    if args.test == 'lighting' or args.test == 'all':
        print("\n=== Running Hand Gesture Lighting Test ===")
        test_gesture_lighting()

    if args.test == 'threshold' or args.test == 'all':
        print("\n=== Running Adaptive Threshold Test ===")
        test_adaptive_threshold()

    if args.test == 'temporal' or args.test == 'all':
        print("\n=== Running Temporal Filtering Test ===")
        test_temporal_filtering()

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
