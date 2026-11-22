-- Useful queries for the report / viva. Run with: sqlite3 studio.db < db/queries.sql

-- 1. All sessions and how many active edits each has
SELECT i.id, i.name, COUNT(t.id) AS active_steps
FROM images i LEFT JOIN transforms_applied t ON t.image_id = i.id AND t.undone = 0
GROUP BY i.id;

-- 2. The active edit chain of image 1, in order, with each determinant
SELECT step_no, name, params, determinant
FROM transforms_applied WHERE image_id = 1 AND undone = 0 ORDER BY step_no;

-- 3. Edits that mirror the image (negative determinant)
SELECT image_id, step_no, name FROM transforms_applied WHERE determinant < 0;

-- 4. Edits that change size, grouped by name (|det| != 1)
SELECT name, COUNT(*) AS uses, AVG(ABS(determinant)) AS avg_area_factor
FROM transforms_applied GROUP BY name;

-- 5. Saved results with their sizes
SELECT r.id, i.name, r.interpolation, r.out_width, r.out_height, r.created_at
FROM results r JOIN images i ON i.id = r.image_id ORDER BY r.id DESC;

-- 6. Redo stack of image 1 (undone edits), next redo first
SELECT step_no, name FROM transforms_applied
WHERE image_id = 1 AND undone = 1 ORDER BY step_no ASC;
