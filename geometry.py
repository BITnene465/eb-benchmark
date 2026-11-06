"""Local geometry: source-defined windows, separate supports, bounded mass transport."""

import math

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter, sobel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree


def rgb(image):
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def clamp(box, size):
    return [
        max(0, math.floor(box[0])),
        max(0, math.floor(box[1])),
        min(size[0], math.ceil(box[2])),
        min(size[1], math.ceil(box[3])),
    ]


def aligned_crop(image, reference_box, candidate_box, window, fillcolor="white"):
    """Align center and area; retain aspect-ratio and local shape errors."""
    size = (window[2] - window[0], window[3] - window[1])
    if candidate_box is None:
        return Image.new(image.mode, size, fillcolor)
    a, b = np.asarray(reference_box, float), np.asarray(candidate_box, float)
    scale = math.sqrt(np.prod(b[2:] - b[:2]) / np.prod(a[2:] - a[:2]))
    offset = (b[:2] + b[2:]) / 2 + scale * (np.asarray(window[:2]) - (a[:2] + a[2:]) / 2)
    return image.transform(
        size,
        Image.Transform.AFFINE,
        (scale, 0, offset[0], 0, scale, offset[1]),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fillcolor,
    )


def boundary_mass(pixels, support, quantiles):
    """RGB total variation / local contrast. Total mass is not normalized."""
    pixels = np.asarray(pixels, float)
    if not support.any():
        return np.zeros(support.shape)
    low, high = np.quantile(pixels[support], quantiles, axis=0)
    contrast = np.linalg.norm(high - low)
    if contrast <= 1 / 255:
        return np.zeros(support.shape)
    gx = np.stack([sobel(pixels[:, :, c], axis=1) for c in range(3)], axis=-1)
    gy = np.stack([sobel(pixels[:, :, c], axis=0) for c in range(3)], axis=-1)
    return np.sqrt(np.sum(gx * gx + gy * gy, axis=-1)) / (8 * contrast) * support


def boundary_mass_local(pixels, support, mode="local", radius=3,
                        fixed_scale=0.1, contrast_floor=2 * math.sqrt(3) / 255):
    """Experimental local evidence; not used by the frozen evaluation entry.

    Fixed mode caps RGB gradient strength at a common 0.1 scale. Local mode
    divides by the RGB range in a radius-3 neighbourhood, with a two-code-value
    contrast floor. An edit cannot change distant weights via a crop statistic.
    Masking is after filtering: the sampling halo is max(1, radius), not zero.
    """
    pixels = np.asarray(pixels, float)
    gx = np.stack([sobel(pixels[:, :, c], axis=1) for c in range(3)], axis=-1)
    gy = np.stack([sobel(pixels[:, :, c], axis=0) for c in range(3)], axis=-1)
    strength = np.sqrt(np.sum(gx * gx + gy * gy, axis=-1)) / 8
    if mode == "fixed":
        scale = fixed_scale
    elif mode == "local":
        size = (2 * radius + 1, 2 * radius + 1, 1)
        colour_range = maximum_filter(pixels, size=size) - minimum_filter(pixels, size=size)
        scale = np.maximum(np.linalg.norm(colour_range, axis=-1), contrast_floor)
    else:
        raise ValueError("Expected fixed or local boundary mode")
    return np.minimum(strength / scale, 1) * support


def _shared_flow(aa, bb, r, c):
    """Bipartite integer capacities; shared by exact and certified bin graphs."""
    if not len(r):
        return 0
    n, m = len(aa), len(bb)
    source, sink = n + m, n + m + 1
    rows = np.r_[np.full(n, source), r, n + np.arange(m)]
    cols = np.r_[np.arange(n), n + c, np.full(m, sink)]
    capacities = np.r_[aa, np.minimum(aa[r], bb[c]), bb]
    graph = csr_matrix((capacities, (rows, cols)), shape=(n + m + 2, n + m + 2))
    return int(maximum_flow(graph, source, sink, method="dinic").flow_value)


def transport(a, b, tolerances, precision=256):
    """Maximum shared mass within each distance tolerance, then mean Dice overlap."""
    qa, qb = [np.floor(x * precision).astype(np.int64) for x in (a, b)]
    x, y = np.argwhere(qa > 0), np.argwhere(qb > 0)
    aa, bb = qa[qa > 0], qb[qb > 0]
    ma, mb = int(aa.sum()), int(bb.sum())
    if max(ma, mb) >= np.iinfo(np.int32).max:
        raise ValueError("Evidence exceeds scipy's integer flow capacity")
    if not ma:
        return {"score": None, "curve": [], "status": "reference_unobservable"}
    if not mb:
        return {"score": 0.0, "curve": [0.0] * len(tolerances), "status": "prediction_missing"}
    distances = cKDTree(x).sparse_distance_matrix(
        cKDTree(y), max(tolerances), output_type="coo_matrix"
    )
    counts = {}
    for tolerance in np.unique(tolerances):
        selected = distances.data <= tolerance
        r, c = distances.row[selected], distances.col[selected]
        counts[float(tolerance)] = _shared_flow(aa, bb, r, c)
    curve = [2 * counts[float(t)] / (ma + mb) for t in tolerances]
    return {
        "score": float(np.mean(curve)),
        "curve": curve,
        "source_mass": ma / precision,
        "candidate_mass": mb / precision,
        "tolerances_px": list(tolerances),
        "status": "measured",
    }


def bounded_transport(a, b, tolerances, precision=256, grids=(32, 16, 8, 4, 2, 1),
                      max_edges=200000, max_flows=40, score_gap=0.005):
    """Certified bounds for the same quantized pixel problem, not downsampling.

    Quantize pixels before binning. A bin pair is fully compatible if the centre
    distance plus both occupied-point radii is <= tolerance, and possibly
    compatible if the distance minus radii is <= tolerance. Max flows bound
    the original flow from below/above. Grid 1 is the exact pixel graph.
    Budgets limit graph construction/flow calls; unresolved bounds stay intervals.
    """
    arrays = [np.asarray(x, float) for x in (a, b)]
    if any(x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() for x in arrays):
        raise ValueError("Boundary fields must be finite nonnegative 2D arrays")
    if precision <= 0 or len(tolerances) == 0 or min(tolerances) < 0:
        raise ValueError("Positive precision and nonnegative tolerances required")
    qa, qb = [np.floor(x * precision).astype(np.int64) for x in arrays]
    ma, mb = int(qa.sum()), int(qb.sum())
    if max(ma, mb) >= np.iinfo(np.int32).max:
        raise ValueError("Evidence exceeds scipy's integer flow capacity")
    if not ma or not mb:
        value = 0.0 if ma else None
        return {"score": value, "score_lower": value, "score_upper": value,
                "curve_lower": [value]*len(tolerances), "curve_upper": [value]*len(tolerances),
                "source_mass": ma/precision, "candidate_mass": mb/precision,
                "status": "prediction_missing" if ma else "reference_unobservable", "history": []}
    points = [np.argwhere(q > 0) for q in (qa, qb)]
    masses = [q[q > 0] for q in (qa, qb)]
    h, w = np.minimum(qa.shape, qb.shape)
    common = int(np.minimum(qa[:h, :w], qb[:h, :w]).sum())
    tau = np.unique(np.asarray(tolerances, float))
    order = np.searchsorted(tau, np.asarray(tolerances, float))
    lower = np.full(len(tau), common, dtype=np.int64)
    upper = np.full(len(tau), min(ma, mb), dtype=np.int64)
    upper[tau < 1] = common  # Integer pixel centres: no distinct pair is <1 apart.
    history, flows = [], 0

    def bins(x, mass, grid):
        labels, inverse = np.unique(x // grid, axis=0, return_inverse=True)
        count = len(labels)
        lo = np.full((count, 2), np.inf); hi = np.full((count, 2), -np.inf)
        np.minimum.at(lo, inverse, x); np.maximum.at(hi, inverse, x)
        capacity = np.zeros(count, np.int64); np.add.at(capacity, inverse, mass)
        centres = (lo + hi) / 2
        radii = np.linalg.norm(hi - lo, axis=1) / 2
        return centres, radii, capacity

    for grid in grids:
        if int(grid) != grid or grid < 1:
            raise ValueError("Grid widths must be positive integers")
        if 2 * np.mean((upper - lower)[order]) / (ma + mb) <= score_gap or flows >= max_flows:
            break
        x, rx, aa = bins(points[0], masses[0], int(grid))
        y, ry, bb = bins(points[1], masses[1], int(grid))
        xt, yt = cKDTree(x), cKDTree(y)
        for k, tolerance in enumerate(tau):
            if lower[k] == upper[k] or flows >= max_flows:
                continue
            search = tolerance + rx.max() + ry.max()
            search = np.nextafter(search, np.inf)
            count = int(xt.count_neighbors(yt, search))
            record = {"grid": int(grid), "tolerance": float(tolerance),
                      "source_bins": len(x), "candidate_bins": len(y), "possible_pairs_envelope": count}
            if count + len(x) + len(y) > max_edges:
                history.append({**record, "state": "edge_budget"}); continue
            distances = xt.sparse_distance_matrix(yt, search, output_type="coo_matrix")
            r, c, distance = distances.row, distances.col, distances.data
            radius = rx[r] + ry[c]
            # Conservatively expand float bounds. Singleton bins use exactly the
            # same centre distances/comparison as the pixel implementation.
            guard = np.where(radius > 0, 8*np.finfo(float).eps*np.maximum(1, distance+radius+tolerance), 0)
            full = distance + radius + guard <= tolerance
            possible = distance - radius - guard <= tolerance
            lo = _shared_flow(aa, bb, r[full], c[full]); flows += 1
            if np.array_equal(full, possible):
                hi = lo
            elif flows < max_flows:
                hi = _shared_flow(aa, bb, r[possible], c[possible]); flows += 1
            else:
                hi = int(upper[k])
            lower[k] = max(lower[k], lo); upper[k] = min(upper[k], hi)
            # M is nondecreasing in tolerance; use this exact property to tighten.
            lower = np.maximum.accumulate(lower)
            upper = np.minimum.accumulate(upper[::-1])[::-1]
            history.append({**record, "state": "computed", "lower_mass": int(lower[k]),
                            "upper_mass": int(upper[k]), "full_pairs": int(full.sum()),
                            "possible_pairs": int(possible.sum())})
    lo_curve = 2*lower[order]/(ma+mb); hi_curve = 2*upper[order]/(ma+mb)
    exact = bool(np.array_equal(lower, upper))
    return {"score": float(lo_curve.mean()) if exact else None,
            "score_lower": float(lo_curve.mean()), "score_upper": float(hi_curve.mean()),
            "curve_lower": lo_curve.tolist(), "curve_upper": hi_curve.tolist(),
            "source_mass": ma/precision, "candidate_mass": mb/precision,
            "tolerances_px": list(tolerances), "status": "measured" if exact else "bounded",
            "flow_calls": flows, "history": history}


def compare_geometry(
    a, b, window_support, source_support, candidate_support, scale, image_size, config
):
    options = config["geometry"]
    fields = [
        boundary_mass(pixels, window_support & support, options["contrast_quantiles"])
        for pixels, support in ((a, source_support), (b, candidate_support))
    ]
    rho = options["raster_floor_px_at_2048"] * max(image_size) / 2048
    tolerances = [rho + v * scale for v in options["tolerance_lambdas"]]
    return transport(*fields, tolerances, options["mass_precision"])


def unit_features(unit, size, config):
    """Convert annotated outlines, corners, tails and partitions to measurement windows."""
    if unit["kind"] != "shape":
        return []
    b = unit["bbox"]
    w, h = b[2] - b[0], b[3] - b[1]
    margin = 3 * max(size) / 2048
    outline = unit.get("outline_reference", {})
    features = [
        {
            "id": unit["id"] + "__outline",
            "family": "outline",
            "window": clamp([b[0] - 6, b[1] - 6, b[2] + 6, b[3] + 6], size),
            "scale": math.hypot(w, h),
            "guide_paths": outline.get("paths", []),
            "guide_width": 2 * config["outline_guide_half_width_px_at_2048"] * max(size) / 2048,
            "excluded_polygons": outline.get("excluded_polygons", []),
        }
    ]
    if (
        outline.get("coverage") != "full_visible_outline"
        or not outline.get("paths")
        or outline.get("status")
        not in ("codex_visual_checked_not_independent_human_GT", "synthetic_source_definition")
    ):
        features[0]["reference_unavailable"] = True
    corners = unit.get("corners", []) + unit.get("body_corners", [])
    if unit.get("radius"):
        r = unit["radius"]
        corners = [
            {
                "type": "round",
                "start": [x + dx * r, y],
                "mid": [x + dx * r * 0.3, y + dy * r * 0.3],
                "end": [x, y + dy * r],
            }
            for x, y, dx, dy in (
                (b[0], b[1], 1, 1),
                (b[2], b[1], -1, 1),
                (b[0], b[3], 1, -1),
                (b[2], b[3], -1, -1),
            )
        ]

    def add(family, suffix, points, pad, guided=False):
        points = np.asarray(points, float)
        lo, hi = points.min(0), points.max(0)
        item = {
            "id": unit["id"] + "__" + suffix,
            "family": family,
            "window": clamp([*(lo - pad), *(hi + pad)], size),
            "scale": max(float(np.linalg.norm(hi - lo)), 1),
        }
        if guided:
            item["guide_paths"] = [points.tolist()]
        features.append(item)

    for i, corner in enumerate(corners):
        if corner.get("type") == "round":
            add("corner", f"corner{i}", [corner[k] for k in ("start", "mid", "end")], margin)
    tail = np.asarray(unit.get("tail", {}).get("points", []), float).reshape(-1, 2)
    if len(tail) >= 2:
        add("tail", "tail", tail, margin, True)
    for i, split in enumerate(unit.get("splits", [])):
        corners = split.get("split_corners", [])
        points = [c["point"] for c in corners if "point" in c]
        points += [
            c[k]
            for c in corners
            if c.get("type") == "round"
            for k in ("start", "mid", "end")
            if k in c
        ]
        if len(points) >= 2:
            add("partition", f"partition{i}", points, margin + 3, True)
    return features


def feature_score(
    reference, candidate, unit, match, feature, config, candidate_support, reference_support
):
    if feature.get("reference_unavailable") or match["status"] == "measurement_unavailable":
        return {"score": None}
    if match["status"] == "prediction_missing":
        return {"score": 0.0}
    if reference_support is None or candidate_support is None:
        return {"score": None}
    window = feature["window"]
    a = rgb(reference.crop(window))
    b = rgb(aligned_crop(candidate, unit["bbox"], match["bbox"], window))
    mask = np.ones(a.shape[:2], np.uint8)
    if feature.get("guide_paths"):
        mask[:] = 0
        paths = [
            np.round(np.asarray(p) - window[:2]).astype(np.int32) for p in feature["guide_paths"]
        ]
        width = max(6, round(feature.get("guide_width", 12 * max(reference.size) / 2048)))
        cv2.polylines(mask, paths, False, 1, width)
    for polygon in feature.get("excluded_polygons", []):
        cv2.fillPoly(mask, [np.round(np.asarray(polygon) - window[:2]).astype(np.int32)], 0)
    observed = np.asarray(
        aligned_crop(Image.fromarray(candidate_support), unit["bbox"], match["bbox"], window, 0),
        bool,
    )
    source = reference_support[window[1] : window[3], window[0] : window[2]]
    evidence = compare_geometry(
        a, b, mask.astype(bool), source, observed, feature["scale"], reference.size, config
    )
    return {"score": evidence["score"], "boundary_evidence": evidence}


def relation_feature(
    reference, candidate, relation, kind, config, candidate_support, reference_support
):
    if reference_support is None or candidate_support is None:
        return None
    points = np.asarray(relation.get("dense_reference_path", relation["points"]), float)
    start, end = points[0], points[-1]
    half_width = relation.get("path_half_width_px", 7)
    head_length, head_width = (
        relation.get("head_length_px", 18),
        relation.get("head_half_width_px", 10),
    )
    if kind == "arrowhead":
        for previous in points[-2::-1]:
            if np.linalg.norm(end - previous) >= 30:
                break
        direction = (end - previous) / max(np.linalg.norm(end - previous), 1)
        normal = np.array([-direction[1], direction[0]])
        polygon = np.array(
            [
                end - direction + normal * head_width,
                end - direction - normal * head_width,
                end - direction * head_length - normal * head_width,
                end - direction * head_length + normal * head_width,
            ]
        )
        box = clamp([*polygon.min(0), *polygon.max(0)], reference.size)
        scale = head_length
    else:
        box = clamp(
            [*(points.min(0) - half_width - 1), *(points.max(0) + half_width + 1)], reference.size
        )
        scale = float(np.linalg.norm(end - start))
    a, b = rgb(reference.crop(box)), rgb(candidate.crop(box))
    mask = np.zeros(a.shape[:2], np.uint8)
    if kind == "arrowhead":
        cv2.fillPoly(mask, [np.round(polygon - box[:2]).astype(np.int32)], 1)
    elif relation.get("path_evidence_kind") == "sampled_points":
        for point in points:
            cv2.circle(mask, tuple(np.round(point - box[:2]).astype(int)), 8, 1, -1)
    else:
        cv2.polylines(
            mask, [np.round(points - box[:2]).astype(np.int32)], False, 1, round(2 * half_width)
        )
    source = reference_support[box[1] : box[3], box[0] : box[2]]
    observed = candidate_support[box[1] : box[3], box[0] : box[2]]
    return compare_geometry(
        a, b, mask.astype(bool), source, observed, scale, reference.size, config
    )["score"]
