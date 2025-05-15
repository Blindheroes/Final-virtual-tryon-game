# Virtual Try-On Game Testing Suite

This folder contains tests for various aspects of the virtual try-on game system.

## Tests Available

1. **Hand Gesture Distance Test** - Tests gesture recognition at different distances
2. **Hand Gesture Lighting Test** - Tests gesture recognition under different lighting conditions
3. **Adaptive Threshold Test** - Tests the effectiveness of adaptive thresholding
4. **Temporal Filtering Test** - Tests the stability of gesture recognition with temporal filtering

## Running Tests

You can run individual tests or all tests using the `run_tests.py` script.

```bash
# Run all tests
python run_tests.py --test all

# Run specific test
python run_tests.py --test distance
python run_tests.py --test lighting
python run_tests.py --test threshold
python run_tests.py --test temporal
```

## Test Features

- All tests have a 5-second visual countdown before starting
- Test results are saved in the `results` folder
- CSV files are generated with timestamp in the filename
