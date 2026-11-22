import numpy as np
import pytest

from app.batch import build_queue, process_queue
from app.imageio import make_sample_image
from app.session import EditorSession
from db.database import Database
from math_engine.matrix import SingularMatrixError

IMG = make_sample_image(60, 40)


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "t.db"))
    yield d
    d.close()


def test_apply_undo_redo_roundtrip(db):
    ed = EditorSession(db, IMG, "t.png")
    ed.apply("Rotate", {"angle": 30})
    ed.apply("Shear", {"kx": 0.3, "ky": 0})
    after_two = ed.render()
    assert ed.undo() and ed.undo() and not ed.undo()
    assert np.array_equal(ed.render(), IMG)               # back to the original
    assert ed.redo() and ed.redo() and not ed.redo()
    assert np.array_equal(ed.render(), after_two)          # redo order is correct

def test_new_edit_after_undo_clears_redo(db):
    ed = EditorSession(db, IMG)
    ed.apply("Rotate", {"angle": 10}); ed.undo()
    assert ed.can_redo()
    ed.apply("Flip horizontal", {})
    assert not ed.can_redo()
    assert len(db.list_steps(ed.image_id)) == 1

def test_zero_determinant_edit_is_rejected(db):
    ed = EditorSession(db, IMG)
    with pytest.raises(SingularMatrixError):
        ed.apply("Scale", {"sx": 1, "sy": 0})
    assert not ed.can_undo() and db.list_steps(ed.image_id) == []

def test_too_large_result_is_rejected(db):
    ed = EditorSession(db, IMG)
    with pytest.raises(ValueError):
        ed.apply("Scale", {"sx": 1000, "sy": 1000})

def test_reopen_session_restores_undo_and_redo_stacks(db):
    ed = EditorSession(db, IMG, "t.png")
    for name, p in [("Rotate", {"angle": 20}), ("Flip vertical", {}), ("Scale", {"sx": 1.2, "sy": 1.2})]:
        ed.apply(name, p)
    ed.undo(); ed.undo()                                   # 1 active, 2 on redo stack
    state = ed.render()
    again = EditorSession.from_db(db, ed.image_id)
    assert len(again.undo_stack) == 1 and len(again.redo_stack) == 2
    assert np.array_equal(again.render(), state)
    again.redo()
    assert again.active_steps()[-1].name == "Flip vertical"   # redo restores oldest undone first

def test_save_result_and_list(db):
    ed = EditorSession(db, IMG)
    ed.apply("Rotate", {"angle": 45})
    rid = ed.save_result()
    rows = db.list_results(ed.image_id)
    assert rows[0]["id"] == rid and rows[0]["out_width"] == ed.render().shape[1]

def test_list_images_counts_active_steps(db):
    ed = EditorSession(db, IMG, "a.png")
    ed.apply("Rotate", {"angle": 5}); ed.apply("Rotate", {"angle": 5}); ed.undo()
    assert db.list_images()[0]["steps"] == 1

def test_batch_queue_is_fifo_and_uses_each_image_centre(db):
    imgs = [("a", IMG), ("b", make_sample_image(30, 30))]
    out = process_queue(build_queue(imgs), [("Flip horizontal", {})], "nearest")
    assert [n for n, _ in out] == ["a", "b"]
    assert np.array_equal(out[0][1], IMG[:, ::-1])
    assert np.array_equal(out[1][1], imgs[1][1][:, ::-1])
