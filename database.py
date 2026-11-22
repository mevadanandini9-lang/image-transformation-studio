"""SQLite access layer for the Image Transformation Studio."""
import json
import os
import sqlite3

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


class Database:
    def __init__(self, path="studio.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- images -------------------------------------------------------
    def add_image(self, name, width, height, mode, png_bytes):
        cur = self.conn.execute(
            "INSERT INTO images(name, width, height, mode, data) VALUES (?,?,?,?,?)",
            (name, width, height, mode, png_bytes))
        self.conn.commit()
        return cur.lastrowid

    def get_image(self, image_id):
        return self.conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()

    def list_images(self):
        """All sessions with the number of active (not undone) steps."""
        return self.conn.execute(
            """SELECT i.id, i.name, i.created_at,
                      COALESCE(SUM(CASE WHEN t.undone = 0 THEN 1 ELSE 0 END), 0) AS steps
               FROM images i LEFT JOIN transforms_applied t ON t.image_id = i.id
               GROUP BY i.id ORDER BY i.id DESC""").fetchall()

    # ---- transforms_applied -------------------------------------------
    def add_step(self, image_id, step_no, name, params, matrix, det):
        cur = self.conn.execute(
            """INSERT INTO transforms_applied
               (image_id, step_no, name, params, matrix, determinant)
               VALUES (?,?,?,?,?,?)""",
            (image_id, step_no, name, json.dumps(params), json.dumps(matrix), det))
        self.conn.commit()
        return cur.lastrowid

    def list_steps(self, image_id):
        return self.conn.execute(
            "SELECT * FROM transforms_applied WHERE image_id=? ORDER BY step_no",
            (image_id,)).fetchall()

    def set_undone(self, step_id, undone):
        self.conn.execute("UPDATE transforms_applied SET undone=? WHERE id=?",
                          (1 if undone else 0, step_id))
        self.conn.commit()

    def delete_undone(self, image_id):
        """A new edit after an undo discards the redo history."""
        self.conn.execute(
            "DELETE FROM transforms_applied WHERE image_id=? AND undone=1", (image_id,))
        self.conn.commit()

    # ---- results ------------------------------------------------------
    def save_result(self, image_id, interpolation, out_w, out_h, matrix, png_bytes):
        cur = self.conn.execute(
            """INSERT INTO results
               (image_id, interpolation, out_width, out_height, matrix, data)
               VALUES (?,?,?,?,?,?)""",
            (image_id, interpolation, out_w, out_h, json.dumps(matrix), png_bytes))
        self.conn.commit()
        return cur.lastrowid

    def list_results(self, image_id):
        return self.conn.execute(
            "SELECT id, interpolation, out_width, out_height, created_at "
            "FROM results WHERE image_id=? ORDER BY id DESC", (image_id,)).fetchall()
