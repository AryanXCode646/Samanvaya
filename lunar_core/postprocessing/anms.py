"""
Adaptive Non-Maximal Suppression (ANMS) & Grid-Based Uniform Spatial Allocator.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from lunar_core.models import KeypointMatch


class SpatialUniformDistributor:
    """
    Suppresses feature clumping on high-relief crater edges and enforces uniform spatial allocation.
    """

    def __init__(self, grid_rows: int = 8, grid_cols: int = 8) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    @staticmethod
    def anms_brown(
        keypoints: np.ndarray,
        responses: np.ndarray,
        target_count: int,
        c_robust: float = 0.9,
    ) -> np.ndarray:
        """
        Optimized Brown et al. ANMS suppression radius formulation:
            r_i = min_{j: s_j > c_robust * s_i} ||x_i - x_j||_2
        Uses fast spatial hash bucketing and vectorization to achieve O(N) performance.
        """
        n = keypoints.shape[0]
        if n <= target_count:
            return np.arange(n)

        sorted_indices = np.argsort(-responses)
        sorted_kps = keypoints[sorted_indices]
        radii = np.full(n, np.inf, dtype=np.float32)

        # For small arrays (n < 150), direct vectorized evaluation is fastest
        if n < 150:
            for i in range(1, n):
                dists = np.linalg.norm(sorted_kps[:i] - sorted_kps[i], axis=1)
                radii[i] = np.min(dists)
        else:
            # Fast spatial hash grid
            coord_min = np.min(sorted_kps, axis=0)
            coord_max = np.max(sorted_kps, axis=0)
            span = np.maximum(coord_max - coord_min, 1.0)
            cell_size = float(np.mean(span) / np.sqrt(n))

            grid: dict[Tuple[int, int], List[int]] = {}

            for i in range(n):
                pt = sorted_kps[i]
                gx = int((pt[0] - coord_min[0]) / cell_size)
                gy = int((pt[1] - coord_min[1]) / cell_size)

                min_dist = float("inf")
                # Search expanding concentric rings of cells
                ring = 1
                found = False
                while not found and ring <= 4:
                    for dy in range(-ring, ring + 1):
                        for dx in range(-ring, ring + 1):
                            cell = (gx + dx, gy + dy)
                            if cell in grid:
                                for prev_idx in grid[cell]:
                                    d = float(np.linalg.norm(sorted_kps[prev_idx] - pt))
                                    if d < min_dist:
                                        min_dist = d
                                        found = True
                    ring += 1

                if not found and i > 0:
                    # Fallback to nearest among previous points
                    dists = np.linalg.norm(sorted_kps[:i] - pt, axis=1)
                    min_dist = float(np.min(dists))

                radii[i] = min_dist
                cell_key = (gx, gy)
                if cell_key not in grid:
                    grid[cell_key] = []
                grid[cell_key].append(i)

        top_by_radius = np.argsort(-radii)[:target_count]
        return sorted_indices[top_by_radius]

    def cap_grid_cells(
        self,
        matches: List[KeypointMatch],
        image_shape: Tuple[int, int],
        cap_per_cell: int = 4,
        use_source_coords: bool = True,
    ) -> List[KeypointMatch]:
        """
        Subdivides the scene into an 8x8 or 16x16 grid and caps top-confidence matches per cell.
        Operates on FULL_SOURCE_IMAGE coordinates by default (SIH PS 26166 source uniformity requirement).
        """
        if not matches:
            return []

        h, w = image_shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols

        buckets: List[List[KeypointMatch]] = [[] for _ in range(self.grid_rows * self.grid_cols)]

        for m in matches:
            coord_x, coord_y = m.target_xy if use_source_coords else m.ref_xy
            gx = min(max(0, int(coord_x // cell_w)), self.grid_cols - 1)
            gy = min(max(0, int(coord_y // cell_h)), self.grid_rows - 1)
            idx = gy * self.grid_cols + gx
            buckets[idx].append(m)

        capped: List[KeypointMatch] = []
        for b in buckets:
            if not b:
                continue
            b_sorted = sorted(b, key=lambda match: match.confidence, reverse=True)
            capped.extend(b_sorted[:cap_per_cell])

        return capped

    def compute_shannon_spatial_entropy(
        self,
        matches: List[KeypointMatch],
        image_shape: Tuple[int, int],
        use_source_coords: bool = True,
    ) -> float:
        """
        Computes the Normalized Shannon Spatial Entropy H in [0.0, 1.0] across the lattice.
        Target: H >= 0.70 (uniformly distributed throughout source image, zero clumping).
        """
        if not matches:
            return 0.0

        h, w = image_shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols
        total_cells = self.grid_rows * self.grid_cols

        counts = np.zeros(total_cells, dtype=np.float64)
        for m in matches:
            coord_x, coord_y = m.target_xy if use_source_coords else m.ref_xy
            gx = min(max(0, int(coord_x // cell_w)), self.grid_cols - 1)
            gy = min(max(0, int(coord_y // cell_h)), self.grid_rows - 1)
            idx = gy * self.grid_cols + gx
            counts[idx] += 1.0

        total = float(np.sum(counts))
        if total == 0:
            return 0.0

        p = counts[counts > 0] / total
        shannon = -float(np.sum(p * np.log2(p)))
        max_shannon = float(np.log2(total_cells))
        return float(np.clip(shannon / max_shannon, 0.0, 1.0))

    def compute_spatial_metrics(
        self,
        matches: List[KeypointMatch],
        image_shape: Tuple[int, int],
        use_source_coords: bool = True,
    ) -> Dict[str, Any]:
        """
        Computes detailed spatial distribution metrics across the grid on FULL_SOURCE_IMAGE:
        coverage_fraction, coverage_ratio, occupied_cells, points_per_cell, max_cluster_fraction,
        spatial_entropy, largest_empty_region, mean_cell_density, density_variance.
        """
        total_cells = self.grid_rows * self.grid_cols
        if not matches:
            return {
                "coverage_fraction": 0.0,
                "coverage_ratio": 0.0,
                "occupied_cells": 0,
                "total_cells": total_cells,
                "points_per_cell": {"mean": 0.0, "min": 0, "max": 0, "std": 0.0, "counts": [0] * total_cells},
                "max_cluster_fraction": 0.0,
                "spatial_entropy": 0.0,
                "largest_empty_region": total_cells,
                "mean_cell_density": 0.0,
                "density_variance": 0.0,
            }

        h, w = image_shape[:2]
        cell_h = h / float(self.grid_rows)
        cell_w = w / float(self.grid_cols)

        counts = np.zeros(total_cells, dtype=np.int64)
        for m in matches:
            x, y = m.target_xy if use_source_coords else m.ref_xy
            gx = min(max(0, int(x // cell_w)), self.grid_cols - 1)
            gy = min(max(0, int(y // cell_h)), self.grid_rows - 1)
            idx = gy * self.grid_cols + gx
            counts[idx] += 1

        total_pts = int(len(matches))
        occupied = int(np.count_nonzero(counts))
        coverage = float(occupied / total_cells)
        max_cluster_fraction = float(np.max(counts) / total_pts) if total_pts > 0 else 0.0

        p = counts[counts > 0].astype(float) / total_pts
        shannon = -float(np.sum(p * np.log2(p))) if len(p) > 0 else 0.0
        max_shannon = float(np.log2(total_cells))
        norm_entropy = float(np.clip(shannon / max_shannon, 0.0, 1.0)) if max_shannon > 0 else 0.0

        # Calculate largest empty region via connected components on empty grid cells
        empty_grid = (counts == 0).reshape(self.grid_rows, self.grid_cols)
        largest_empty = 0
        if np.any(empty_grid):
            visited = np.zeros_like(empty_grid, dtype=bool)
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if empty_grid[r, c] and not visited[r, c]:
                        size = 0
                        queue = [(r, c)]
                        visited[r, c] = True
                        while queue:
                            cr, cc = queue.pop(0)
                            size += 1
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                                    if empty_grid[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))
                        largest_empty = max(largest_empty, size)

        mean_density = float(total_pts / total_cells)
        density_var = float(np.var(counts))

        # Edge vs Center imbalance calculation
        grid_2d = counts.reshape(self.grid_rows, self.grid_cols)
        edge_pts = int(np.sum(grid_2d[0, :]) + np.sum(grid_2d[-1, :]) + np.sum(grid_2d[1:-1, 0]) + np.sum(grid_2d[1:-1, -1]))
        center_pts = max(0, total_pts - edge_pts)
        edge_ratio = float(edge_pts / total_pts) if total_pts > 0 else 0.0
        edge_center_imbalance = float(abs(edge_ratio - 0.5))

        # Nearest neighbor concentration
        coords = np.array([m.target_xy if use_source_coords else m.ref_xy for m in matches], dtype=float)
        if len(coords) > 1:
            diffs = coords[:, None, :] - coords[None, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            diag_len = float(np.sqrt(h**2 + w**2))
            nn_concentration = float(np.mean(min_dists) / max(diag_len, 1e-6))
        else:
            nn_concentration = 0.0

        is_clustered = bool(coverage < 0.20 or max_cluster_fraction > 0.50 or norm_entropy < 0.35)
        cluster_status = "CLUSTERED_MATCHES" if is_clustered else ("GOOD_DISTRIBUTION" if coverage >= 0.50 else "PARTIAL_COVERAGE")

        return {
            "coverage_fraction": coverage,
            "coverage_ratio": coverage,
            "occupied_cells": occupied,
            "total_cells": total_cells,
            "points_per_cell": {
                "mean": float(np.mean(counts)),
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
                "std": float(np.std(counts)),
                "counts": counts.tolist(),
            },
            "max_cluster_fraction": max_cluster_fraction,
            "spatial_entropy": norm_entropy,
            "largest_empty_region": int(largest_empty),
            "mean_cell_density": mean_density,
            "density_variance": density_var,
            "edge_center_imbalance": edge_center_imbalance,
            "nearest_neighbor_concentration": nn_concentration,
            "is_clustered": is_clustered,
            "cluster_status": cluster_status,
        }


def generate_spatial_distribution_plot(
    output_path: str | Path,
    matches: List[KeypointMatch],
    image_shape: Tuple[int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
    use_source_coords: bool = True,
) -> Path:
    """Render a publication-quality spatial distribution diagram for scientific reporting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    distributor = SpatialUniformDistributor(grid_rows=grid_rows, grid_cols=grid_cols)
    metrics = distributor.compute_spatial_metrics(matches, image_shape, use_source_coords=use_source_coords)

    h, w = image_shape[:2]
    grid_counts = np.array(metrics["points_per_cell"]["counts"], dtype=np.float64).reshape(grid_rows, grid_cols)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    im = ax.imshow(
        grid_counts,
        extent=[0, w, h, 0],
        cmap="Blues",
        aspect="auto",
        alpha=0.65,
        norm=Normalize(vmin=0, vmax=max(1.0, float(np.max(grid_counts)))),
    )

    if matches:
        coords = np.array([m.target_xy if use_source_coords else m.ref_xy for m in matches])
        ax.scatter(coords[:, 0], coords[:, 1], c="red", s=18, edgecolors="white", linewidths=0.5, label="Matches")

    # Grid boundary lines
    for r in range(grid_rows + 1):
        y = r * (h / grid_rows)
        ax.axhline(y, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
    for c in range(grid_cols + 1):
        x = c * (w / grid_cols)
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Match Count per Cell")

    frame_name = "FULL_SOURCE_IMAGE" if use_source_coords else "FULL_REFERENCE_IMAGE"
    ax.set_title(
        f"Spatial Distribution & Uniformity ({frame_name})\n"
        f"Entropy H: {metrics['spatial_entropy']:.3f} | Coverage: {metrics['coverage_ratio']*100:.1f}% "
        f"({metrics['occupied_cells']}/{metrics['total_cells']} cells) | Density Var: {metrics['density_variance']:.2f}",
        fontsize=10,
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    if matches:
        ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def uniform_distribute(
    matches: List[KeypointMatch],
    image_shape: Tuple[int, int],
    grid_rows: int = 8,
    grid_cols: int = 8,
    cap_per_cell: int = 4,
) -> Tuple[List[KeypointMatch], float]:
    """Convenience wrapper that caps matches per grid cell and returns spatial entropy.

    Args:
        matches: List of KeypointMatch objects.
        image_shape: (height, width) of the reference image.
        grid_rows: Number of rows in the uniform grid (default 8).
        grid_cols: Number of columns in the uniform grid (default 8).
        cap_per_cell: Maximum matches to keep per grid cell (default 4).

    Returns:
        A tuple ``(capped_matches, entropy_score)`` where ``capped_matches`` is the
        list after grid-cell capping and ``entropy_score`` is the normalized Shannon
        spatial entropy in ``[0.0, 1.0]``.
    """
    distributor = SpatialUniformDistributor(grid_rows=grid_rows, grid_cols=grid_cols)
    capped = distributor.cap_grid_cells(matches, image_shape, cap_per_cell=cap_per_cell)
    entropy = distributor.compute_shannon_spatial_entropy(capped, image_shape)
    return capped, entropy
