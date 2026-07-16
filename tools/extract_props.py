"""Cut foreground props out of the v7 level plates with an alpha channel.

Background removal = flood-fill from the crop border over "background" pixels
(luminance below THRESH). Bright prop outlines act as barriers, so a prop's
darker interior stays opaque as long as its silhouette is reasonably enclosed.
This is the same trick slice_sprites.py uses to drop panel borders.

Usage: extract_props.py            (extract all)
       extract_props.py <name>     (single, prints coverage)
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

# name -> (src plate, bbox x0,y0,x1,y1, background luminance threshold)
# boxes tightened to a single object so little neighbouring art is captured.
PROPS = {
    "idea":       ("idea-level-v2.png",       (902, 546, 1004, 648), 52),  # one lightbulb box
    "code":       ("code-level-v2.png",       (28,  520, 108,  628), 40),  # potted plant
    "validation": ("validation-level-v2.png", (896, 500, 1098, 626), 44),  # 3 check-tanks
    "review":     ("review-level-v2.png",     (356, 470, 442,  566), 40),  # review bot
    "production": ("production-level-v2.png",  (892, 556, 1004, 650), 62),  # crate stack
}


def extract(name):
    src, (x0, y0, x1, y1), thresh = PROPS[name]
    img = Image.open(f"assets/raw/{src}").convert("RGB").crop((x0, y0, x1, y1))
    a = np.asarray(img).astype(int)
    H, W, _ = a.shape
    lum = a.mean(axis=2)
    bg = lum < thresh                       # candidate background pixels

    # 1) flood-fill background inward from the border (bright outlines block it)
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    reached = ndimage.binary_propagation(border & bg, mask=bg)
    keep = ~reached                         # opaque = not reachable background

    # 2) fill interior holes, then drop small speckle (open)
    keep = ndimage.binary_fill_holes(keep)
    keep = ndimage.binary_opening(keep, iterations=1)

    # 3) keep only components that touch the central column (drop edge chunks)
    lbl, n = ndimage.label(keep)
    cx0, cx1 = int(W * 0.25), int(W * 0.75)
    central = set(np.unique(lbl[:, cx0:cx1])) - {0}
    keep = np.isin(lbl, list(central))
    keep = ndimage.binary_fill_holes(keep)
    keep = ndimage.binary_erosion(keep, iterations=1)     # shave the dark bg rim

    alpha = Image.fromarray((keep * 255).astype("uint8"))  # crisp edges for pixel art
    out = img.convert("RGBA"); out.putalpha(alpha)
    out.save(f"assets/{name}-prop.png")
    print(f"{name:11} {out.size}  opaque={keep.mean()*100:.0f}%  -> assets/{name}-prop.png")


for n in ([sys.argv[1]] if len(sys.argv) > 1 else PROPS):
    extract(n)
