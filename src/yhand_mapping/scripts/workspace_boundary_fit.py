#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult2D:
    x_grid: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    point_count: int


@dataclass
class FitResult3D:
    x_grid: np.ndarray
    z_grid: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    point_count: int


def _smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.size < 3:
        return arr.copy()
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(padded, kernel, mode="valid")


def _smooth_2d(arr: np.ndarray, passes: int = 1) -> np.ndarray:
    out = arr.copy()
    for _ in range(max(1, int(passes))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def _interp_clamped(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    if x <= float(x_grid[0]):
        return float(y_grid[0])
    if x >= float(x_grid[-1]):
        return float(y_grid[-1])
    return float(np.interp(x, x_grid, y_grid))


def fit_boundary_2d(
    points_xy: np.ndarray,
    max_outward_expand: float,
    grid_size: int = 80,
) -> FitResult2D:
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if pts.shape[0] < 8:
        x = pts[:, 0]
        y = pts[:, 1]
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), max(2, min(16, pts.shape[0])))
        y_min = np.full_like(x_grid, float(np.min(y)))
        y_max = np.full_like(x_grid, float(np.max(y)))
        return FitResult2D(x_grid=x_grid, y_min=y_min, y_max=y_max, point_count=int(pts.shape[0]))

    x = pts[:, 0]
    y = pts[:, 1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1e-6

    n_grid = max(24, int(grid_size))
    x_grid = np.linspace(x_min, x_max, n_grid)
    step = (x_max - x_min) / max(1, n_grid - 1)
    window = max(step * 2.5, 1e-6)

    raw_low = np.empty(n_grid, dtype=np.float64)
    raw_high = np.empty(n_grid, dtype=np.float64)

    for i, x0 in enumerate(x_grid):
        mask = np.abs(x - x0) <= window
        if np.count_nonzero(mask) < 6:
            order = np.argsort(np.abs(x - x0))
            pick = max(6, min(30, len(order)))
            ys = y[order[:pick]]
        else:
            ys = y[mask]
        raw_low[i] = float(np.min(ys))
        raw_high[i] = float(np.max(ys))

    low = _smooth_1d(raw_low, window=5)
    high = _smooth_1d(raw_high, window=5)

    low = np.minimum(low, raw_low)
    high = np.maximum(high, raw_high)

    ext = abs(float(max_outward_expand))
    low = np.maximum(low, raw_low - ext)
    high = np.minimum(high, raw_high + ext)

    # Ensure all points are inside by local correction.
    for _ in range(2):
        changed = False
        for xi, yi in pts:
            lo_i = _interp_clamped(x_grid, low, float(xi))
            hi_i = _interp_clamped(x_grid, high, float(xi))
            j = int(np.argmin(np.abs(x_grid - float(xi))))
            if yi < lo_i:
                low[j] = min(low[j], float(yi))
                changed = True
            if yi > hi_i:
                high[j] = max(high[j], float(yi))
                changed = True
        if not changed:
            break

    low = np.minimum(low, high - 1e-9)
    return FitResult2D(
        x_grid=x_grid,
        y_min=low,
        y_max=high,
        point_count=int(pts.shape[0]),
    )


def fit_boundary_3d(
    points_xyz: np.ndarray,
    max_outward_expand: float,
    x_grid_size: int = 30,
    z_grid_size: int = 24,
) -> FitResult3D:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if pts.shape[0] < 20:
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xg = np.linspace(float(np.min(x)), float(np.max(x)), max(2, min(12, pts.shape[0])))
        zg = np.linspace(float(np.min(z)), float(np.max(z)), max(2, min(12, pts.shape[0])))
        ylo = np.full((xg.size, zg.size), float(np.min(y)), dtype=np.float64)
        yhi = np.full((xg.size, zg.size), float(np.max(y)), dtype=np.float64)
        return FitResult3D(x_grid=xg, z_grid=zg, y_min=ylo, y_max=yhi, point_count=int(pts.shape[0]))

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1e-6
    if abs(z_max - z_min) < 1e-9:
        z_max = z_min + 1e-6

    nx = max(12, int(x_grid_size))
    nz = max(10, int(z_grid_size))
    xg = np.linspace(x_min, x_max, nx)
    zg = np.linspace(z_min, z_max, nz)
    dx = (x_max - x_min) / max(1, nx - 1)
    dz = (z_max - z_min) / max(1, nz - 1)
    rx = max(dx * 1.5, 1e-6)
    rz = max(dz * 1.5, 1e-6)

    raw_low = np.empty((nx, nz), dtype=np.float64)
    raw_high = np.empty((nx, nz), dtype=np.float64)

    xz = np.stack([x, z], axis=1)
    for ix, x0 in enumerate(xg):
        for iz, z0 in enumerate(zg):
            mask = (np.abs(x - x0) <= rx) & (np.abs(z - z0) <= rz)
            if np.count_nonzero(mask) < 8:
                dist = np.linalg.norm(xz - np.array([x0, z0], dtype=np.float64), axis=1)
                order = np.argsort(dist)
                pick = max(8, min(40, len(order)))
                ys = y[order[:pick]]
            else:
                ys = y[mask]
            raw_low[ix, iz] = float(np.min(ys))
            raw_high[ix, iz] = float(np.max(ys))

    low = _smooth_2d(raw_low, passes=2)
    high = _smooth_2d(raw_high, passes=2)

    low = np.minimum(low, raw_low)
    high = np.maximum(high, raw_high)

    ext = abs(float(max_outward_expand))
    low = np.maximum(low, raw_low - ext)
    high = np.minimum(high, raw_high + ext)

    # Ensure all points are inside by nearest-cell correction.
    for _ in range(2):
        changed = False
        for xi, yi, zi in pts:
            ix = int(np.argmin(np.abs(xg - float(xi))))
            iz = int(np.argmin(np.abs(zg - float(zi))))
            if yi < low[ix, iz]:
                low[ix, iz] = float(yi)
                changed = True
            if yi > high[ix, iz]:
                high[ix, iz] = float(yi)
                changed = True
        if not changed:
            break

    low = np.minimum(low, high - 1e-9)
    return FitResult3D(
        x_grid=xg,
        z_grid=zg,
        y_min=low,
        y_max=high,
        point_count=int(pts.shape[0]),
    )


def projected_points_stats(points: np.ndarray):
    if points.size == 0:
        return {
            "count": 0,
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        }

    out = {
        "count": int(points.shape[0]),
        "x_min": float(np.min(points[:, 0])),
        "x_max": float(np.max(points[:, 0])),
        "y_min": float(np.min(points[:, 1])),
        "y_max": float(np.max(points[:, 1])),
    }
    if points.shape[1] >= 3:
        out["z_min"] = float(np.min(points[:, 2]))
        out["z_max"] = float(np.max(points[:, 2]))
    else:
        out["z_min"] = None
        out["z_max"] = None
    return out


def to_jsonable_float_array(arr: np.ndarray):
    return [float(v) for v in np.asarray(arr, dtype=np.float64).tolist()]


def to_jsonable_float_matrix(arr: np.ndarray):
    mat = np.asarray(arr, dtype=np.float64)
    return [[float(v) for v in row] for row in mat.tolist()]
