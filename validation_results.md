# Validation of from-scratch code against numpy

| n | abs error in det | max abs error in inverse | max abs error in A@B |
|---|---|---|---|
| 2 | 0.00e+00 | 1.11e-16 | 2.22e-16 |
| 3 | 1.11e-16 | 2.22e-16 | 2.22e-16 |
| 5 | 1.39e-17 | 3.20e-14 | 4.44e-16 |
| 8 | 7.11e-14 | 5.55e-16 | 4.44e-16 |

Zero-determinant demo: det = 0: the image is flattened onto a line/point. Information is lost and the matrix is not invertible.

Forward mapping holes (rotate 30, scale 1.5): 96001 of 172801 pixels; inverse mapping: 0.
