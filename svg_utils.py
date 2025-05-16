from svgpathtools import svg2paths, wsvg, Path, Line
from shapely.geometry import Polygon, box, LineString
from shapely.ops import unary_union, clip_by_rect
import numpy as np

#pip install svgpathtools shapely==2.0.2  # shapely < 2.0 uses GEOS inside; 2.x is pure Py
 
# ------------------------------------------------------------------
# CONFIGURATION – match these to your robot units
# ------------------------------------------------------------------
SVG_IN      = "original.svg"       # what you already have
SVG_OUT     = "original_with_hatch.svg"

STROKE_WIDTH = 3.0                 # mm – how “fat” a painted line is
HATCH_SPACING = 10.0               # mm between hatch lines
HATCH_ANGLE   = 45                 # °  0 = horizontal, 90 = vertical

CANVAS_W, CANVAS_H = 600, 600      # same as self.svg_width/height
# ------------------------------------------------------------------


# 1)  read the original svg ---------------------------------------------------
paths, attr = svg2paths(SVG_IN)

# 2)  convert every path into a Shapely polygon (buffer = stroke width) -------
painted = []
for p in paths:
    # sample path into many tiny straight segments (svgpathtools helper)
    pts = [p.point(t) for t in np.linspace(0, 1, 500)]
    poly = LineString([(z.real, z.imag) for z in pts]).buffer(STROKE_WIDTH/2)
    painted.append(poly)

painted_area = unary_union(painted)

# 3)  blank area  = canvas  \ painted_area ------------------------------------
canvas_poly   = box(0, 0, CANVAS_W, CANVAS_H)
blank_area    = canvas_poly.difference(painted_area)

if blank_area.is_empty:
    print("No empty space left – nothing to hatch.")
    exit()

# 4)  generate a hatch pattern over the whole canvas --------------------------
angle_rad = np.deg2rad(HATCH_ANGLE)
direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

# estimate how many lines we need to cover the canvas diagonal
diag = np.linalg.norm([CANVAS_W, CANVAS_H])
num_lines = int(diag / HATCH_SPACING) + 2

hatch_paths = []
for i in range(-num_lines, num_lines):
    # point on an infinite line:  p = origin + i * spacing * perpendicular(dir)
    offset = np.array([-direction[1], direction[0]]) * i * HATCH_SPACING
    p0 = -direction * diag + offset
    p1 =  direction * diag + offset
    line = LineString([tuple(p0), tuple(p1)])

    # clip that long line to the blank polygon
    clipped = line.intersection(blank_area)
    if clipped.is_empty:
        continue
    if clipped.geom_type == 'LineString':
        hatch_paths.append(clipped)
    elif clipped.geom_type == 'MultiLineString':
        hatch_paths.extend(list(clipped))

# 5)  convert hatch LineStrings back to svgpathtools Path objects -------------
new_svg_paths = paths[:]                 # start with originals
for ln in hatch_paths:
    (x0, y0), (x1, y1) = list(ln.coords)
    new_svg_paths.append(Path(Line(complex(x0, y0), complex(x1, y1))))

# 6)  write out a new SVG ------------------------------------------------------
wsvg(new_svg_paths, filename=SVG_OUT,
     attributes=[{"stroke": "black", "fill": "none",
                  "stroke-width": str(STROKE_WIDTH)}]*len(new_svg_paths))

print("Wrote", SVG_OUT, "with", len(hatch_paths), "extra hatch strokes.")
