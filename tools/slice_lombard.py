"""Split the Lombard Street plate into a parallax backdrop and a ground strip.

The plate's bottom band is a full-width brick wall with a grass lip — that
grass line becomes the game's walkable ground. Everything above it is saved
as the backdrop; a clean segment of the wall becomes a repeating ground tile.
"""
import numpy as np
from PIL import Image

SRC = "assets/raw/Lombardi Street Level.png"
img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(int)
H, W, _ = a.shape
r, g, b = a[..., 0], a[..., 1], a[..., 2]

# Row-by-row "grass" fraction in the bottom third
green = (g > r + 15) & (g > b + 15)
frac = green.mean(axis=1)
lo = int(H * 0.6)
rows = [(y, frac[y]) for y in range(lo, H)]
print("green fraction profile (bottom third, every 6px):")
for y, f in rows[::6]:
    print(f"  y={y} ({y/H:.3f})  {f:.2f}  {'#' * int(f * 40)}")

# Grass band = lowest contiguous run of rows with frac > 0.45
band = frac > 0.45
grass_rows = [y for y in range(lo, H) if band[y]]
assert grass_rows, "no full-width grass band found"
# take the lowest cluster (walkable lip of the bottom wall)
clusters = [[grass_rows[0]]]
for y in grass_rows[1:]:
    if y - clusters[-1][-1] <= 3:
        clusters[-1].append(y)
    else:
        clusters.append([y])
grass_top = clusters[-1][0]
print(f"\ngrass line: y={grass_top} ({grass_top/H:.4f} of height)")

# Backdrop: everything above the grass line
bg = img.crop((0, 0, W, grass_top))
bg.save("assets/lombard-bg.png")
print(f"backdrop: {bg.size} -> assets/lombard-bg.png")

# Ground tile: clean wall segment (avoid garage at left, structures at right)
x0, x1 = 1180, 1468
tile = img.crop((x0, grass_top, x1, H))
tile.save("assets/lombard-ground.png")
print(f"ground tile: {tile.size} -> assets/lombard-ground.png")

# Sky colors for the zone gradient behind/around the plate
sky_mask = (b > r + 30) & (b > 120)
top_rows = a[0:10][sky_mask[0:10]]
mid_y = int(H * 0.35)
mid_rows = a[mid_y:mid_y + 10][sky_mask[mid_y:mid_y + 10]]
def hexcol(px_arr):
    m = np.median(px_arr, axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*m)
print(f"sky top ~ {hexcol(top_rows)}   sky mid ~ {hexcol(mid_rows)}")
# wall base color for the fill below the tile
wall = a[H - 8:H, x0:x1].reshape(-1, 3)
print(f"wall base ~ {hexcol(wall)}")
