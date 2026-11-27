# Image Transformation Studio

Linear Algebra Project 6 - School of Engineering, JG University (BTech CSE)

## Problem statement
Photo apps rotate, flip, shear and resize pictures. A digital image is a grid of
pixels and every one of these edits moves pixels according to a **matrix**. Moving
each input pixel forward leaves holes, so professional tools go backwards: for each
**output** pixel they use the **inverse matrix** to find the input pixel it came from.

This app lets a user upload a grayscale or colour image, apply rotation (any angle),
horizontal/vertical flip, shear and scaling, see before/after, undo/redo, and save
the whole chain of edits in a database so a session can be reopened.

## Setup
```bash
pip install -r requirements.txt
streamlit run app/app.py          # start the studio
pytest                            # run the tests (48)
python demo_outputs.py            # regenerate figures + validation table
```
Run every command from the project root.

## Repository structure
| Folder | Contents |
| --- | --- |
| `math_engine/` | **From scratch**: matrix multiply, determinant, inverse, transformation matrices, composition, inverse mapping with nearest/bilinear interpolation |
| `db/` | `schema.sql` (3 tables), `database.py` (SQLite), `queries.sql`, `sample_data/` |
| `ds/` | `Stack` (undo/redo) and `FifoQueue` (batch processing), both on linked lists |
| `app/` | Streamlit UI (`app.py`), editor session logic, batch processing, image I/O |
| `tests/` | unit tests + `validation_results.md` |
| `report/` | `REPORT_TEMPLATE.md` and `figures/` for the final PDF report |

## Where Linear Algebra is used
| Concept | Where in the code | Why |
| --- | --- | --- |
| Linear transformations / matrices | `transforms.py` | Each edit is a 3x3 matrix (homogeneous coordinates) acting on [x, y, 1] |
| Matrix multiplication | `matrix.mat_mul`, `transforms.compose` | A chain of edits = a product of matrices; order matters |
| Inverse | `matrix.inverse`, `warp.inverse_warp` | Output pixel -> input pixel without holes; undo = multiply by inverse |
| Determinant | `matrix.determinant`, `transforms.describe_determinant` | Sign = flip, magnitude = area scaling, zero = collapse (not invertible) |

## Design notes
* The **original image is never modified**. All edits are multiplied into one matrix and the original is
  warped once, so quality does not degrade with each edit.
* Every edit is applied about the **image centre** (translate to origin, apply, translate back).
* An edit with det = 0 is rejected (it would destroy information and has no inverse).
* Undo uses a **Stack**. Redo also uses a Stack, because redo must restore the most recently undone edit
  first (a FIFO queue would restore them in the wrong order). The **Queue** is used for batch processing.
* Pillow is used only to read, write and downscale image files (uploads are limited to 1024 px on the long side).

## Screenshots
Add screenshots of the app here after running it (Studio tab, "Why inverse mapping?" tab, "Zero determinant" tab).
Required-output figures are already in `report/figures/`.


