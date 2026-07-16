"""Split a painted level plate into a parallax backdrop and a ground strip.

Usage: slice_plate.py <name>   (name in CONFIGS)

Auto-detects the walkable grass line unless grass_top is pinned in the
config (needed when the plate has water gaps or multi-level terraces that
break the full-width grass band).
"""
import sys
import numpy as np
from PIL import Image

CONFIGS = {
    # v7 pipeline-stage levels (Joshua, 2026-07-16) — all 1908×824, new taller art.
    # Output to *2-*.png so v6's frozen assets are untouched.
    "idea2":       {"src": "assets/raw/idea-level-v2.png",       "out": "idea2",       "tile_x": (1600, 1900), "grass_top": 668},
    "code2":       {"src": "assets/raw/code-level-v2.png",       "out": "code2",       "tile_x": (720, 1020),  "grass_top": 632},
    "validation2": {"src": "assets/raw/validation-level-v2.png", "out": "validation2", "tile_x": (1280, 1580), "grass_top": 620},
    "review2":     {"src": "assets/raw/review-level-v2.png",     "out": "review2",     "tile_x": (0, 300),     "grass_top": 608},
    "production2": {"src": "assets/raw/production-level-v2.png",  "out": "production2", "tile_x": (600, 900),   "grass_top": 644},

    # v4 pipeline-stage levels (Joshua, 2026-07-15) — all 2698×583, same layout
    "idea":       {"src": "assets/raw/idea-level.png",       "out": "idea",       "tile_x": (1578, 1878), "grass_top": 489},
    "code":       {"src": "assets/raw/code-level.png",       "out": "code",       "tile_x": (618, 918),   "grass_top": 454},
    "validation": {"src": "assets/raw/validation-level.png", "out": "validation", "tile_x": (1315, 1615), "grass_top": 449},
    "review":     {"src": "assets/raw/review-level.png",     "out": "review",     "tile_x": (2119, 2419), "grass_top": 457},
    "production": {"src": "assets/raw/production-level.png",  "out": "production", "tile_x": (2310, 2610), "grass_top": 471},
    "aideck": {
        "src": "assets/raw/amplitude final office.png",
        "out": "aideck",
        "tile_x": (1520, 1800),
        "grass_top": 528,   # glowing orange baseboard of the night deck
    },
    "office": {
        "src": "assets/raw/amplitude indoor office.png",
        "out": "office",
        "tile_x": (985, 1195),
        "grass_top": 715,   # top of the orange floor trim (indoor level, no grass)
    },
    "lombard": {
        "src": "assets/raw/Lombardi Street Level.png",
        "out": "lombard",
        "tile_x": (1180, 1468),
    },
    "goldengate": {
        "src": "assets/raw/golden gate level.png",
        "out": "goldengate",
        "tile_x": (60, 340),
        "grass_top": 612,
    },
    "paintedladies": {
        "src": "assets/raw/painted ladies level.png",
        "out": "paintedladies",
        "tile_x": (110, 390),
        "grass_top": 541,
    },
}

cfg = CONFIGS[sys.argv[1]]
img = Image.open(cfg["src"]).convert("RGB")
a = np.asarray(img).astype(int)
H, W, _ = a.shape
r, g, b = a[..., 0], a[..., 1], a[..., 2]
green = (g > r + 15) & (g > b + 15)
frac = green.mean(axis=1)

grass_top = cfg.get("grass_top")
if grass_top is None:
    print("green fraction profile (bottom 45%, every 8px):")
    for y in range(int(H * 0.55), H, 8):
        print(f"  y={y} ({y/H:.3f})  {frac[y]:.2f}  {'#' * int(frac[y] * 40)}")
    band = [y for y in range(int(H * 0.55), H) if frac[y] > 0.45]
    if band:
        clusters = [[band[0]]]
        for y in band[1:]:
            (clusters[-1].append(y) if y - clusters[-1][-1] <= 3 else clusters.append([y]))
        grass_top = clusters[-1][0]
        print(f"auto grass line: y={grass_top} ({grass_top/H:.4f})")
    else:
        print("no full-width grass band found — pin grass_top in the config")
        sys.exit(1)

if cfg.get("tile_x") is None:
    print("tile_x not set — inspect the plate and pin a clean wall segment")
    sys.exit(0)

x0, x1 = cfg["tile_x"]
bg = img.crop((0, 0, W, grass_top))
bg.save(f"assets/{cfg['out']}-bg.png")
tile = img.crop((x0, grass_top, x1, H))
tile.save(f"assets/{cfg['out']}-ground.png")
print(f"backdrop {bg.size} -> assets/{cfg['out']}-bg.png")
print(f"ground tile {tile.size} -> assets/{cfg['out']}-ground.png")

sky_mask = (b > r + 30) & (b > 120)
def hexcol(p): m = np.median(p, axis=0).astype(int); return "#{:02x}{:02x}{:02x}".format(*m)
print("sky top ~", hexcol(a[0:10][sky_mask[0:10]]))
mid = int(H * 0.35)
print("sky mid ~", hexcol(a[mid:mid+10][sky_mask[mid:mid+10]]))
wall = a[H-8:H, x0:x1].reshape(-1, 3)
print("wall base ~", hexcol(wall))
