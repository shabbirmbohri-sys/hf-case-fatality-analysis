"""
Render the Getis-Ord Gi* local clustering map for HF in-hospital case fatality.
No geopandas/shapely available in this sandbox -- TopoJSON (us-atlas, already
Albers-projected) is decoded by hand and drawn with matplotlib PathPatch/Polygon.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA = f"{BASE}/data"
FIGURES = f"{BASE}/figures"
os.makedirs(FIGURES, exist_ok=True)

# ---------- 1. Load analytic sample FIPS order + Gi* values ----------
with open(f"{BASE}/data/analytic_sample_v2_final.csv") as f:
    rows = list(csv.DictReader(f))
fips_order = [r["FIPS"] for r in rows]  # same row order as primary_fit.npz / inference_results.npz

inf = np.load(f"{BASE}/results/inference_results.npz")
Gi = inf["Gi"]
assert len(Gi) == len(fips_order) == 486

cluster = {}
for fips, gi in zip(fips_order, Gi):
    if gi > 1.96:
        cluster[fips] = "high"
    elif gi < -1.96:
        cluster[fips] = "low"
    else:
        cluster[fips] = "ns"

analytic_fips = set(fips_order)

# ---------- 2. Decode TopoJSON (pure Python, no external deps) ----------
# Source: us-atlas (MIT license), https://github.com/topojson/us-atlas
# Re-downloadable from https://cdn.jsdelivr.net/npm/us-atlas@3/counties-albers-10m.json
with open(f"{DATA}/counties-albers-10m.json") as f:
    topo = json.load(f)

scale = topo["transform"]["scale"]
translate = topo["transform"]["translate"]

def decode_arc(arc):
    """Delta-decode + rescale a single TopoJSON arc to (x,y) pairs."""
    x, y = 0, 0
    pts = []
    for dx, dy in arc:
        x += dx
        y += dy
        pts.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
    return pts

raw_arcs = topo["arcs"]
decoded_arcs = [decode_arc(a) for a in raw_arcs]

def arc_coords(idx):
    """Resolve a TopoJSON arc index (possibly negative = reversed) to coordinates."""
    if idx >= 0:
        return decoded_arcs[idx]
    else:
        return list(reversed(decoded_arcs[~idx]))

def ring_coords(arc_indices):
    """Stitch a list of arc indices into one closed ring."""
    coords = []
    for a in arc_indices:
        pts = arc_coords(a)
        if coords and coords[-1] == pts[0]:
            coords.extend(pts[1:])
        else:
            coords.extend(pts)
    return coords

def geometry_to_paths(geom):
    """Return a list of matplotlib Path objects (one per polygon, holes included via
    compound path) for a TopoJSON Polygon or MultiPolygon geometry."""
    paths = []
    if geom["type"] == "Polygon":
        polys = [geom["arcs"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["arcs"]
    else:
        return paths
    for poly in polys:
        all_verts = []
        all_codes = []
        for ring in poly:
            coords = ring_coords(ring)
            all_verts.extend(coords)
            all_codes.append(Path.MOVETO)
            all_codes.extend([Path.LINETO] * (len(coords) - 2))
            all_codes.append(Path.CLOSEPOLY)
            all_verts[-1] = all_verts[0]  # CLOSEPOLY vertex ignored by mpl but keep consistent
        paths.append(Path(all_verts, all_codes))
    return paths

counties_geoms = topo["objects"]["counties"]["geometries"]
states_geoms = topo["objects"]["states"]["geometries"]

# ---------- 3. Build patches ----------
color_map = {"high": "#c0392b", "low": "#2166ac", "ns": "#ececec"}
label_map = {"high": "Significant high cluster (Gi* > 1.96)",
             "low": "Significant low cluster (Gi* < -1.96)",
             "ns": "Not significant / not in analytic sample"}

fig, ax = plt.subplots(figsize=(13, 8.2))

bg_patches = []
fg_patches_by_cat = {"high": [], "low": [], "ns": []}

for geom in counties_geoms:
    fips = geom.get("id")
    paths = geometry_to_paths(geom)
    if fips not in analytic_fips:
        for p in paths:
            bg_patches.append(PathPatch(p))
    else:
        cat = cluster[fips]
        for p in paths:
            fg_patches_by_cat[cat].append(PathPatch(p))

# non-analytic counties: light neutral fill (context, not part of the 486-county sample)
ax.add_collection(PatchCollection(bg_patches, facecolor="#f7f7f7", edgecolor="#d9d9d9", linewidth=0.15, zorder=1))

# analytic, not-significant counties
ax.add_collection(PatchCollection(fg_patches_by_cat["ns"], facecolor=color_map["ns"], edgecolor="#bbbbbb", linewidth=0.2, zorder=2))

# analytic, significant clusters (drawn last so they stand out)
ax.add_collection(PatchCollection(fg_patches_by_cat["low"], facecolor=color_map["low"], edgecolor="#0b3d61", linewidth=0.3, zorder=3))
ax.add_collection(PatchCollection(fg_patches_by_cat["high"], facecolor=color_map["high"], edgecolor="#6e1c11", linewidth=0.3, zorder=4))

# state outlines on top for geographic reference
state_patches = []
for geom in states_geoms:
    for p in geometry_to_paths(geom):
        state_patches.append(PathPatch(p))
ax.add_collection(PatchCollection(state_patches, facecolor="none", edgecolor="#555555", linewidth=0.5, zorder=5))

ax.set_xlim(0, 975)
ax.set_ylim(600, 0)
ax.set_aspect("equal")
ax.axis("off")

from matplotlib.patches import Patch as LegendPatch
legend_elems = [
    LegendPatch(facecolor=color_map["high"], edgecolor="#6e1c11", label=f"Significant high cluster (n={sum(v=='high' for v in cluster.values())})"),
    LegendPatch(facecolor=color_map["low"], edgecolor="#0b3d61", label=f"Significant low cluster (n={sum(v=='low' for v in cluster.values())})"),
    LegendPatch(facecolor=color_map["ns"], edgecolor="#bbbbbb", label=f"Analytic sample, not significant (n={sum(v=='ns' for v in cluster.values())})"),
    LegendPatch(facecolor="#f7f7f7", edgecolor="#d9d9d9", label="Not in analytic sample"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=2, fontsize=9.5, frameon=False, columnspacing=1.4, handletextpad=0.6)

ax.set_title(
    "Figure 1. Local clustering of heart failure in-hospital case fatality\n"
    "Getis-Ord Gi* on residuals from the fully adjusted spatial error model (n = 486 counties)",
    fontsize=11.5, loc="left"
)

plt.tight_layout()
plt.savefig(f"{FIGURES}/Figure1_GetisOrd_map.png", dpi=220, bbox_inches="tight")
print("Saved Figure1_GetisOrd_map.png")
print("Counts:", {k: sum(v == k for v in cluster.values()) for k in ["high", "low", "ns"]})
