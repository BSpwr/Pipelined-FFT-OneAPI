import numpy as np
import sys


def run(numpoints):
    # Tolerance for comparison
    rtol = 1e-04
    atol = 1e-07

    # Calculate numpy fft
    x = [i for i in range(0, numpoints)]
    v = np.fft.fft(x)

    # Example input:
    # {120, 0}, {-8, 19.3137}, {-8, 40.2187}, {-8, 11.9728}
    # {-8, 0}, {-8, -3.31371}, {-8, -1.5913}, {-8, -5.34543}
    # {-8, 8}, {-8, 3.31371}, {-8, 5.34543}, {-8, 1.5913}
    # {-8, -8}, {-8, -19.3137}, {-8, -11.9728}, {-8, -40.2187}
    results = np.empty(0)

    for line in sys.stdin:  # Read input piped from oneAPI program
        for i, item in enumerate(line.split('}')):  # Split by }
            #print(item)
            item = item.lstrip(',')
            item = item.lstrip(' ')
            item = item.lstrip('{')
            if item.isspace() or len(item) == 0:  # Ignore trailing item
                continue
            parts = item.split(',')
            if len(parts) < 2 or parts[0].isspace() or len(parts[0]) == 0:
                continue
            # Parse string into float parts
            real = float(parts[0])
            imag = float(parts[1])
            #print(real, imag)
            results = np.append(results, real + (imag * 1j))  # Add item to array

    print("Expected:", v)
    print("Actual:", results)
    if np.allclose(results, v, rtol, atol):  # Compare two arrays with tolerance
        print("Passed!")
    else:
        print("Failed!")
        difference = np.subtract(results, v)
        print("Difference between actual and expected:", difference)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Must specify number of points")
        exit(-1)
    numpoints = int(sys.argv[1])
    run(numpoints)
