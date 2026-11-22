"""Editor session: original image + chain of edits + undo/redo + database.

Design: the ORIGINAL image is never modified. The chain of edits is multiplied
into ONE matrix and the original is warped once (inverse mapping). That gives
better quality (one interpolation instead of many) and makes undo trivial:
removing the last matrix is the same as multiplying the total by its inverse.
"""
import json
from dataclasses import dataclass

from chain import step_matrix
from imageio import from_png_bytes, to_png_bytes
from ds.stack import Stack
from matrix import SingularMatrixError, determinant
from transforms import compose
from warp import inverse_warp, output_canvas

@dataclass
class Step:
    name: str
    params: dict
    matrix: list
    det: float
    db_id: int
    step_no: int


class EditorSession:
    def __init__(self, db, original, name="image", image_id=None,
                 interpolation="bilinear", expand=True):
        self.db, self.original, self.name = db, original, name
        self.interpolation, self.expand = interpolation, expand
        self.undo_stack, self.redo_stack = Stack(), Stack()
        h, w = original.shape[:2]
        self.width, self.height = w, h
        if image_id is None:
            mode = "L" if original.ndim == 2 else "RGB"
            image_id = db.add_image(name, w, h, mode, to_png_bytes(original))
        self.image_id = image_id

    # ---- state ---------------------------------------------------------
    def active_steps(self):
        return self.undo_stack.items()

    def chain(self):
        return [(s.name, s.params) for s in self.active_steps()]

    def total_matrix(self):
        return compose([s.matrix for s in self.active_steps()])

    def can_undo(self):
        return not self.undo_stack.is_empty()

    def can_redo(self):
        return not self.redo_stack.is_empty()

    # ---- actions -------------------------------------------------------
    def apply(self, name, params):
        M = step_matrix(name, params, self.width, self.height)
        det = determinant(M)
        if abs(det) < 1e-9:
            raise SingularMatrixError(
                f"'{name}' has determinant {det:.4g}: it flattens the image and "
                "cannot be undone, so it is rejected.")
        candidate = compose([s.matrix for s in self.active_steps()] + [M])
        output_canvas(candidate, self.width, self.height, self.expand)  # size check
        self.db.delete_undone(self.image_id)
        self.redo_stack.clear()
        step_no = len(self.undo_stack) + 1
        db_id = self.db.add_step(self.image_id, step_no, name, params, M, det)
        self.undo_stack.push(Step(name, params, M, det, db_id, step_no))

    def undo(self):
        if not self.can_undo():
            return False
        step = self.undo_stack.pop()
        self.db.set_undone(step.db_id, True)
        self.redo_stack.push(step)
        return True

    def redo(self):
        if not self.can_redo():
            return False
        step = self.redo_stack.pop()
        self.db.set_undone(step.db_id, False)
        self.undo_stack.push(step)
        return True

    # ---- output --------------------------------------------------------
    def render(self):
        return inverse_warp(self.original, self.total_matrix(),
                            self.interpolation, self.expand)

    def save_result(self):
        out = self.render()
        h, w = out.shape[:2]
        return self.db.save_result(self.image_id, self.interpolation, w, h,
                                   self.total_matrix(), to_png_bytes(out))

    # ---- reopen a saved session -----------------------------------------
    @classmethod
    def from_db(cls, db, image_id, interpolation="bilinear", expand=True):
        row = db.get_image(image_id)
        original = from_png_bytes(row["data"])
        ed = cls(db, original, row["name"], image_id, interpolation, expand)
        rows = db.list_steps(image_id)

        def to_step(r):
            return Step(r["name"], json.loads(r["params"]), json.loads(r["matrix"]),
                        r["determinant"], r["id"], r["step_no"])

        for r in rows:                       # active edits, oldest first
            if r["undone"] == 0:
                ed.undo_stack.push(to_step(r))
        for r in reversed(rows):             # undone edits: lowest step_no ends on top,
            if r["undone"] == 1:             # so redo restores them in the right order
                ed.redo_stack.push(to_step(r))
        return ed
