"""
Build row-standardized first-order queen contiguity spatial weights matrix,
restricted to the 486-county analytic sample.
Source: 2023 Census county adjacency file (County Name|GEOID|Neighbor Name|Neighbor GEOID)
"""
import os
import csv
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

# Load analytic sample FIPS list, in a fixed order
with open(f"{BASE}/data/analytic_sample_v2_final.csv") as f:
    r = csv.DictReader(f)
    sample = list(r)

fips_list = [row["FIPS"] for row in sample]
n = len(fips_list)
fips_index = {f: i for i, f in enumerate(fips_list)}
print(f"Analytic sample: {n} counties")

# Parse adjacency file
adj_pairs = set()
with open(f"{BASE}/data/raw/county_adjacency2023.txt", encoding="latin-1") as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 4:
            continue
        county_name, county_geoid, neighbor_name, neighbor_geoid = parts[0], parts[1], parts[2], parts[3]
        county_geoid = county_geoid.strip()
        neighbor_geoid = neighbor_geoid.strip()
        if not county_geoid:
            continue
        if county_geoid == neighbor_geoid:
            continue  # exclude self-loop
        if county_geoid in fips_index and neighbor_geoid in fips_index:
            adj_pairs.add((county_geoid, neighbor_geoid))

print(f"Adjacency pairs within sample (directed): {len(adj_pairs)}")

# Build binary adjacency matrix (symmetrize: queen contiguity should be symmetric,
# but the file may not list every pair in both directions consistently -- force symmetry)
W_bin = np.zeros((n, n), dtype=np.float64)
for a, b in adj_pairs:
    i, j = fips_index[a], fips_index[b]
    W_bin[i, j] = 1.0
    W_bin[j, i] = 1.0  # enforce symmetry

np.fill_diagonal(W_bin, 0.0)

neighbor_counts = W_bin.sum(axis=1)
n_islands = int((neighbor_counts == 0).sum())
print(f"Mean within-sample neighbors: {neighbor_counts.mean():.2f}")
print(f"Max within-sample neighbors: {int(neighbor_counts.max())}")
print(f"Counties with zero within-sample neighbors (islands): {n_islands}")

island_fips = [fips_list[i] for i in range(n) if neighbor_counts[i] == 0]
island_info = [(row["FIPS"], row["CountyName"], row["State"]) for row in sample if row["FIPS"] in island_fips]
print("Island counties:", island_info)

# Row-standardize (islands get a row of all zeros -- handled by W_bin row sum being 0;
# division guarded below)
W = np.zeros_like(W_bin)
for i in range(n):
    s = neighbor_counts[i]
    if s > 0:
        W[i, :] = W_bin[i, :] / s

np.save(f"{BASE}/data/W_matrix.npy", W)
with open(f"{BASE}/data/fips_order.csv", "w") as f:
    f.write("FIPS\n")
    for fp in fips_list:
        f.write(fp + "\n")

print("Saved W_matrix.npy and fips_order.csv")
print("W row sums (should be 1 or 0 for islands):", np.unique(np.round(W.sum(axis=1), 6))[:5])
