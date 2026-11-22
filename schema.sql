PRAGMA foreign_keys = ON;

-- One row per uploaded image. A "session" = an image plus its chain of edits.
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    width      INTEGER NOT NULL,
    height     INTEGER NOT NULL,
    mode       TEXT    NOT NULL CHECK (mode IN ('L', 'RGB')),
    data       BLOB    NOT NULL,                 -- original image as PNG bytes
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- The chain of edits, in order. undone = 1 means it sits on the redo stack.
CREATE TABLE IF NOT EXISTS transforms_applied (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    step_no     INTEGER NOT NULL,
    name        TEXT    NOT NULL,                -- Rotate, Flip horizontal, ...
    params      TEXT    NOT NULL,                -- JSON, e.g. {"angle": 30}
    matrix      TEXT    NOT NULL,                -- JSON 3x3 matrix of this step
    determinant REAL    NOT NULL,
    undone      INTEGER NOT NULL DEFAULT 0 CHECK (undone IN (0, 1)),
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (image_id, step_no)
);

-- Rendered outputs the user chose to save.
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id      INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    interpolation TEXT    NOT NULL CHECK (interpolation IN ('nearest', 'bilinear')),
    out_width     INTEGER NOT NULL,
    out_height    INTEGER NOT NULL,
    matrix        TEXT    NOT NULL,              -- combined 3x3 matrix (JSON)
    data          BLOB    NOT NULL,              -- result as PNG bytes
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
