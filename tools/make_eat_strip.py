"""Build the cookie-eating strip from the 5x3 'eat data' sheet.

Keeps only the hat-consistent frames (rows 1-2, indices 0-9): reach under
hat, reveal cookie, bring it down, and three bites. The baked 'data nom nom'
text and soft shadows are dropped by the keep-largest-component pass.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = "assets/raw/datamonster-eat.png"
OUT = "assets/datamonster-eat.png"
PLAY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # row-major indices to keep, in order

img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(int)
H, W, _ = a.shape
lum = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000

dark = lum < 185
labels, n = ndimage.label(dark, structure=np.ones((3, 3)))
comps = []
for sl in ndimage.find_objects(labels):
    h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if h >= 200 and w >= 150:
        comps.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
print(f"{len(comps)} monster components found")
assert len(comps) == 15, "expected a 5x3 grid"

comps.sort(key=lambda b: (b[1] + b[3]) / 2)
rows = [comps[0:5], comps[5:10], comps[10:15]]
order = []
for row in rows:
    order += sorted(row, key=lambda b: b[0])

def extract(box, pad=8):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(W, x1 + pad), min(H, y1 + pad)
    crop = a[y0:y1, x0:x1]
    clum = lum[y0:y1, x0:x1]
    sat = crop.max(axis=2) - crop.min(axis=2)
    light = (clum > 200) | ((clum > 165) & (sat < 25))
    bg_labels, _ = ndimage.label(light, structure=np.ones((3, 3)))
    edge_ids = set(bg_labels[0, :]) | set(bg_labels[-1, :]) | \
               set(bg_labels[:, 0]) | set(bg_labels[:, -1])
    edge_ids.discard(0)
    alpha = np.where(np.isin(bg_labels, list(edge_ids)), 0, 255)
    op_labels, _ = ndimage.label(alpha > 0, structure=np.ones((3, 3)))
    if op_labels.max() > 1:
        sizes = ndimage.sum(alpha > 0, op_labels, range(1, op_labels.max() + 1))
        keep = int(np.argmax(sizes)) + 1
        alpha = np.where(op_labels == keep, alpha, 0)
    # reseal eye whites the edge flood ate through a thin fur gap
    m = ndimage.binary_closing(alpha > 0, iterations=3)
    m = ndimage.binary_fill_holes(m)
    alpha = np.where(m, 255, 0)
    rgba = np.dstack([crop, alpha]).astype(np.uint8)
    im = Image.fromarray(rgba)
    return im.crop(im.getbbox())

frames = [extract(order[i]) for i in PLAY]
fw = max(f.width for f in frames) + 4
fh = max(f.height for f in frames) + 4
strip = Image.new("RGBA", (fw * len(frames), fh), (0, 0, 0, 0))
for i, f in enumerate(frames):
    strip.paste(f, (i * fw + (fw - f.width) // 2, fh - f.height), f)
strip = strip.resize((strip.width // 2, strip.height // 2), Image.NEAREST)
strip.save(OUT)
print(f"{len(frames)} frames, {fw//2}x{fh//2} each -> {OUT}")
