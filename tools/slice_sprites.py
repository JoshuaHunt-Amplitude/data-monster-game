"""Slice the Data Monster sprite sheet into per-animation horizontal strips.

Detects sprite blobs as dark connected components, groups them into the four
labeled rows (standing/walking/running/flying), removes the light background
via edge flood-fill (so white eyes survive), and bottom-center-anchors every
frame onto a uniform per-animation canvas.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def reseal(opaque, k=2):
    """Recover light inlets (eye whites) the edge flood ate through a thin fur
    gap: close necks up to ~2k px wide, then fill the now-enclosed holes. Leg
    and arm gaps are far wider than the eye necks, so they survive untouched.
    Returns a uint8 alpha array."""
    m = ndimage.binary_closing(opaque, iterations=k)
    m = ndimage.binary_fill_holes(m)
    return np.where(m, 255, 0)


SRC = "assets/raw/Data Monster Sprite Sheet.png"
OUT = {
    "idle": "assets/datamonster-idle.png",
    "walk": "assets/datamonster-walk.png",
    "run":  "assets/datamonster-run.png",
    "fly":  "assets/datamonster-fly.png",
}
EXPECTED = {"idle": 3, "walk": 4, "run": 6, "fly": 4}

img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(int)
H, W, _ = a.shape
lum = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000

# Dark-ish pixels = candidate sprite material (monster body, hat, outlines)
dark = lum < 185

labels, n = ndimage.label(dark, structure=np.ones((3, 3)))
comps = []
for sl in ndimage.find_objects(labels):
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    if h >= 70 and w >= 70:  # skip text glyphs, rules, borders
        comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))

# Merge overlapping/nearby boxes (speed lines, detached limbs)
def merge(boxes, gap=18):
    boxes = boxes[:]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            b = boxes.pop()
            for i, o in enumerate(out):
                if not (b[2] + gap < o[0] or o[2] + gap < b[0] or
                        b[3] + gap < o[1] or o[3] + gap < b[1]):
                    out[i] = (min(b[0], o[0]), min(b[1], o[1]),
                              max(b[2], o[2]), max(b[3], o[3]))
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
        if not changed:
            return boxes

comps = merge(comps)
# Group into rows by vertical center
comps.sort(key=lambda b: (b[1] + b[3]) / 2)
rows = []
for b in comps:
    cy = (b[1] + b[3]) / 2
    if rows and abs(cy - rows[-1]["cy"]) < 90:
        rows[-1]["boxes"].append(b)
        rows[-1]["cy"] = np.mean([(x[1] + x[3]) / 2 for x in rows[-1]["boxes"]])
    else:
        rows.append({"cy": cy, "boxes": [b]})

assert len(rows) == 4, f"expected 4 animation rows, got {len(rows)}"
order = ["idle", "walk", "run", "fly"]

def extract(box, pad=6):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = a[y0:y1, x0:x1]
    clum = lum[y0:y1, x0:x1]
    # Background = light or gray pixels connected to the crop edge (covers the
    # panel borders and baked ground shadows, which are low-saturation grays;
    # the white eyes are enclosed by the body so the edge flood never reaches them)
    sat = crop.max(axis=2) - crop.min(axis=2)
    # light bg + panel borders + baked gray ground shadows; the flood below
    # only removes what's reachable from the frame edge, and the character's
    # dark outline (lum < 90) blocks it from entering the body
    light = (clum > 200) | ((clum > 88) & (sat < 45))
    bg_labels, _ = ndimage.label(light, structure=np.ones((3, 3)))
    edge_ids = set(bg_labels[0, :]) | set(bg_labels[-1, :]) | \
               set(bg_labels[:, 0]) | set(bg_labels[:, -1])
    edge_ids.discard(0)
    bg = np.isin(bg_labels, list(edge_ids))
    alpha = reseal(~bg)
    rgba = np.dstack([crop, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")

for name, row in zip(order, rows):
    boxes = sorted(row["boxes"], key=lambda b: b[0])
    assert len(boxes) == EXPECTED[name], \
        f"{name}: expected {EXPECTED[name]} frames, got {len(boxes)}"
    frames = [extract(b) for b in boxes]
    fw = max(f.width for f in frames) + 4
    fh = max(f.height for f in frames) + 4
    strip = Image.new("RGBA", (fw * len(frames), fh), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        strip.paste(f, (i * fw + (fw - f.width) // 2, fh - f.height), f)
    strip.save(OUT[name])
    print(f"{name}: {len(frames)} frames, {fw}x{fh} each -> {OUT[name]}")
