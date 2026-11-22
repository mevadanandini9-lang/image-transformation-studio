# Image Transformation Studio - Final Report (template)
Convert to PDF before submission. Write the explanations in YOUR OWN words.

## 1. Problem statement and real-world context
Photo editors, games and CAD tools move pixels with matrices. Forward mapping leaves holes; inverse mapping does not.

## 2. Mathematical model
* Pixel (x, y) -> column vector [x, y, 1]^T (homogeneous coordinates)
* Rotation (counter-clockwise on screen, y axis down): [[cos t, sin t, 0], [-sin t, cos t, 0], [0, 0, 1]]
* Flip horizontal diag(-1, 1, 1); flip vertical diag(1, -1, 1)
* Shear [[1, kx, 0], [ky, 1, 0], [0, 0, 1]]; scale diag(sx, sy, 1)
* Edit about centre c: T(c) . M . T(-c). Chain: M_total = M_n ... M_2 M_1
* Assumptions: ...

## 3. Algorithms implemented and why they work
Matrix multiply, determinant (elimination, pivoting), inverse (Gauss-Jordan), inverse mapping, bilinear interpolation.
(One paragraph each. Show a hand-solved 3x3 example.)

## 4. Database design
ER diagram: images 1--* transforms_applied ; images 1--* results. Describe each column (see `db/schema.sql`).

## 5. Data structures and complexity
Stack push/pop O(1); Queue enqueue/dequeue O(1); warp O(output pixels).

## 6. Results, validation and limitations
Insert `figures/1_chain_and_undo.png`, `2_zero_determinant.png`, `3_inverse_mapping_no_holes.png`
and `tests/validation_results.md`. Limitations: no translation edit in the UI, upload downscaled to 1024 px, etc.

## 7. Linear Algebra concept map
| Concept | Where used | Why |
| --- | --- | --- |
| Linear transformation | each edit | moves every pixel by a matrix |
| Matrix multiplication | chain of edits | combines edits; order matters |
| Inverse | inverse mapping, undo | no holes; reverses an edit |
| Determinant | edit classification | flip / scale / collapse |
| Homogeneous coordinates | centre-based edits | translation becomes a matrix too |

## "Where is Linear Algebra used here and why does it work?" (one per phase)
Phase 1 (matrices) / Phase 2 (inverse mapping) / Phase 3 (determinant) / Phase 4 (composition and undo): ...

## 8. Team contribution table and references
| Member | Role (rotates at Review 1) | Contribution |
| --- | --- | --- |
