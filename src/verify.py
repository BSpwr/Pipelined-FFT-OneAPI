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
    results = np.empty(len(v))
    for line in sys.stdin:  # Read input piped from oneAPI program
        if line.isspace() or len(line) == 0:  # Ignore empty lines
            continue
        for item in line.split('}'):  # Split by }
            if item.isspace() or len(item) == 0:  # Ignore trailing item
                continue
            parts = item.split(',')
            parts[0] = parts[0][1:]  # Remove first {, now just number
            
            # Parse string into float parts
            real = float(parts[0])
            imag = float(parts[1])
            np.append(results, real+(imag*1j))  # Add item to array
    
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