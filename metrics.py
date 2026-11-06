"""F score mappings, hierarchical aggregation, appearance and text evidence."""

import math
import unicodedata
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from geometry import aligned_crop, clamp, rgb


def half_score(error, half_error):
    if error < 0 or half_error <= 0:
        raise ValueError("Expected nonnegative error and positive scale")
    return 2 ** (-error / half_error)


def layout_score(reference_box, candidate_box, position_half_error=0.1, size_half_ratio=1.1):
    if candidate_box is None:
        return 0.0
    a, b = np.asarray(reference_box, float), np.asarray(candidate_box, float)
    aw, ah = a[2:] - a[:2]
    bw, bh = b[2:] - b[:2]
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    pos = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2) / math.hypot(aw, ah)
    size = (abs(math.log(bw / aw)) + abs(math.log(bh / ah))) / 2
    return (half_score(pos, position_half_error) + half_score(size, math.log(size_half_ratio))) / 2


def content_score(reference, candidate):
    """NFC plus nonsemantic whitespace; preserve punctuation and scientific symbols."""

    def norm(s):
        return " ".join(unicodedata.normalize("NFC", s).split())

    a, b = norm(reference), norm(candidate)
    if not a:
        raise ValueError("A text requirement must have a nonempty reference")
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return max(0.0, 1 - previous[-1] / len(a))


def srgb_to_lab(value):
    """D65 sRGB [0,1] to CIELAB, for visible color samples in a common space."""
    value = np.asarray(value, dtype=float)
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError("Expected RGB in [0,1]")
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = (linear @ matrix.T) / np.array([0.95047, 1, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack(
        [
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def color_score(reference_rgb, candidate_rgb, half_error=10):
    delta_e = float(np.linalg.norm(srgb_to_lab(reference_rgb) - srgb_to_lab(candidate_rgb)))
    return half_score(delta_e, half_error)


def color_coverage(reference_rgb, candidate_rgb, weights, half_error=10, known=None):
    """Candidate local color metric; arrays must already share a coordinate frame.

    ``weights`` is a fixed, source-defined attribute mask (optionally soft at
    antialiased boundaries), not the candidate's visible support. Unknown pixels
    retain their budget and give an interval; they are never silently dropped.
    This experimental function does not replace the frozen appearance scorer.
    """
    a, b, w = np.asarray(reference_rgb), np.asarray(candidate_rgb), np.asarray(weights, float)
    if a.shape != b.shape or a.shape[:-1] != w.shape or a.shape[-1] != 3:
        raise ValueError("Expected aligned RGB arrays and same-size weights")
    if half_error <= 0 or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Expected nonnegative finite weights and positive color scale")
    total = float(w.sum())
    if total == 0:
        return {"score": None, "status": "reference_unobservable"}
    known = np.ones(w.shape, bool) if known is None else np.asarray(known, bool)
    if known.shape != w.shape:
        raise ValueError("Known-pixel mask must match the source weights")
    k = w * known
    observed_mass = float(k.sum())
    delta = np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)
    response = np.exp2(-delta / half_error)
    integral = float(np.sum(k * response))
    unknown = total - observed_mass
    return {
        "score": integral / total if unknown == 0 else None,
        "lower": integral / total,
        "upper": (integral + unknown) / total,
        "status": "measured" if unknown == 0 else "partially_observed",
        "reference_mass": total,
        "observed_mass": observed_mass,
        "response_integral": integral,
        "unknown_fraction": unknown / total,
        "mean_deltaE76": float(np.sum(k * delta) / observed_mass) if observed_mass else None,
    }


def source_fill_weights(reference, unit, reference_support, excluded_support=None,
                        color_tolerance=0.035, interior_margin=8):
    """Provisional flat-fill mask from source evidence only, never prediction error.

    ``excluded_support`` marks source text, foreground objects and connections.
    This deliberately preserves the old color-selection/margin parameters for
    the aggregation ablation. Gradient/material fills require explicit property
    masks and must not be described as fully measured by this flat-fill mask.
    """
    pixels = rgb(reference)
    yy, xx = np.indices(pixels.shape[:2])
    x0, y0, x1, y1 = unit["bbox"]
    mask = ((xx > x0 + interior_margin) & (xx < x1 - interior_margin)
            & (yy > y0 + interior_margin) & (yy < y1 - interior_margin))
    mask &= np.asarray(reference_support, bool)
    mask &= np.linalg.norm(pixels - hex_rgb(unit["fill"]), axis=-1) < color_tolerance
    if excluded_support is not None:
        mask &= ~np.asarray(excluded_support, bool)
    return mask


def stroke_score(reference_visible_width, candidate_visible_width, half_ratio=1.25):
    if reference_visible_width <= 0:
        raise ValueError("Reference must have a visible stroke")
    if candidate_visible_width <= 0:
        return 0.0
    return half_score(
        abs(math.log(candidate_visible_width / reference_visible_width)),
        math.log(half_ratio),
    )


def macro_score(leaves):
    """GT-fixed family → region → unit → feature budgets, independent of image area."""
    cells = defaultdict(list)
    unknown_families = set()
    for leaf in leaves:
        if leaf.get("state") == "not_applicable":
            continue
        if leaf.get("state") != "prediction_missing" and (
            leaf.get("state") == "unknown" or leaf.get("score") is None
        ):
            unknown_families.add(leaf["family"])
            continue
        value = 0.0 if leaf.get("state") == "prediction_missing" else float(leaf["score"])
        if not 0 <= value <= 1:
            raise ValueError("Leaf scores must be in [0,1]")
        cells[(leaf["family"], leaf["region"], leaf["unit"])].append(value)
    regions = defaultdict(list)
    for (family, region, unit), values in cells.items():
        regions[(family, region)].append(float(np.mean(values)))
    families = defaultdict(list)
    for (family, region), values in regions.items():
        families[family].append(float(np.mean(values)))
    scores = {family: float(np.mean(values)) for family, values in families.items()}
    for family in unknown_families:
        scores[family] = None
    if unknown_families:
        return {
            "score": None,
            "status": "unknown_reference_or_measurement",
            "families": scores,
        }
    return {
        "score": float(np.mean(list(scores.values()))) if scores else None,
        "status": "computed_on_declared_scope" if scores else "no_applicable_requirements",
        "families": scores,
    }


def visual_total(branches, extra_visible_budget=0.0, weights=None):
    if extra_visible_budget < 0:
        raise ValueError("Extra budget cannot be negative")
    if any(branches.get(b) is None for b in ("L", "G", "A", "T")):
        return None
    weights = weights or {b: 0.25 for b in ("L", "G", "A", "T")}
    if abs(sum(weights.values()) - 1) > 1e-8 or any(v < 0 for v in weights.values()):
        raise ValueError("Invalid F branch weights")
    return (
        100
        * sum(weights[b] * branches[b] for b in ("L", "G", "A", "T"))
        / (1 + extra_visible_budget)
    )


def hex_rgb(value):
    return np.array([int(value.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def contrast_core_color(colors, fill_color):
    lab = srgb_to_lab(colors)
    contrast = np.minimum(
        np.linalg.norm(lab - srgb_to_lab(fill_color), axis=1),
        np.linalg.norm(lab - srgb_to_lab([1, 1, 1]), axis=1),
    )
    return np.median(colors[contrast >= np.quantile(contrast, 0.94)], axis=0)


def unit_appearance(
    reference,
    candidate,
    unit,
    match,
    config=None,
    candidate_support=None,
    reference_support=None,
):
    if match["status"] == "prediction_missing":
        return [
            {"family": f, "score": 0.0}
            for f in ["fill", "stroke", "stroke_width"]
            if unit.get({"fill": "fill", "stroke": "stroke", "stroke_width": "stroke_width"}[f])
            is not None
        ]
    if match["bbox"] is None:
        return [
            {"family": f, "score": None}
            for f in ["fill", "stroke", "stroke_width"]
            if unit.get(f) is not None
        ]
    options = config["appearance"]
    box = unit["bbox"]
    window = clamp([box[0] - 7, box[1] - 7, box[2] + 7, box[3] + 7], reference.size)
    a = rgb(reference.crop(window))
    b = rgb(aligned_crop(candidate, box, match["bbox"], window))
    observed_support = np.ones(a.shape[:2], bool)
    source_support = (
        np.ones(a.shape[:2], bool)
        if reference_support is None
        else reference_support[window[1] : window[3], window[0] : window[2]]
    )
    if candidate_support is not None:
        observed_support = np.asarray(
            aligned_crop(
                Image.fromarray(candidate_support),
                box,
                match["bbox"],
                window,
                fillcolor=0,
            ),
            dtype=bool,
        )
    yy, xx = np.indices(a.shape[:2])
    x, y = xx + window[0], yy + window[1]
    frame = np.minimum.reduce(
        [np.abs(x - box[0]), np.abs(x - box[2]), np.abs(y - box[1]), np.abs(y - box[3])]
    )
    interior = (x > box[0] + 8) & (x < box[2] - 8) & (y > box[1] + 8) & (y < box[3] - 8)
    result = []
    stroke_data = None
    for key, region in [("fill", interior), ("stroke", frame <= 7)]:
        if not unit.get(key):
            continue
        color = hex_rgb(unit[key])
        mask = (np.linalg.norm(a - color, axis=2) < 0.035) & region & source_support
        if mask.sum() < 3:
            result.append(
                {
                    "family": key,
                    "score": None,
                    "reason": "reference color not identifiable in rendered evidence",
                }
            )
            continue
        observed_mask = mask & observed_support
        if not observed_mask.any():
            result.append(
                {
                    "family": key,
                    "score": None,
                    "reason": "assigned instance has no observable attribute support",
                }
            )
            continue
        observed = np.median(b[observed_mask], axis=0)
        reference_color = color
        reference_color = np.median(a[mask], axis=0)
        if key == "stroke":
            # Estimate the stroke's core color from the boundary band, not one exact
            # antialiased sample grid. Subpixel raster shifts must not turn blue into fill.
            colors = b[region & observed_support]
            if not len(colors):
                result.append(
                    {
                        "family": key,
                        "score": None,
                        "reason": "assigned instance has no visible boundary support",
                    }
                )
                continue
            fill_color = (
                np.median(b[interior & observed_support], axis=0)
                if (interior & observed_support).any()
                else np.array([1.0, 1.0, 1.0])
            )
            observed = contrast_core_color(colors, fill_color)
            ref_fill_color = (
                np.median(a[interior & source_support], axis=0)
                if (interior & source_support).any()
                else np.array([1.0, 1.0, 1.0])
            )
            reference_color = contrast_core_color(a[region & source_support], ref_fill_color)
        result.append(
            {
                "family": key,
                "score": color_score(
                    reference_color, observed, options.get("deltaE76_half_error", 10)
                ),
                "reference_rgb": reference_color.tolist(),
                "nominal_GT_rgb": color.tolist(),
                "reference_color_mode": options.get("reference_color_mode", "nominal_GT"),
                "observed_rgb": observed.tolist(),
                "pixels": int(mask.sum()),
            }
        )
        if key == "stroke":
            stroke_data = (reference_color, observed, region)
    if unit.get("stroke_width") is not None and stroke_data:
        if not (interior & source_support).any() or not (interior & observed_support).any():
            result.append(
                {
                    "family": "stroke_width",
                    "score": None,
                    "reason": "source_or_candidate_background_support_unavailable",
                }
            )
            return result
        ref_color, pred_color, region = stroke_data
        trim = max(12, 0.2 * min(box[2] - box[0], box[3] - box[1]))
        vertical = (
            ((np.abs(x - box[0]) <= 7) | (np.abs(x - box[2]) <= 7))
            & (y > box[1] + trim)
            & (y < box[3] - trim)
        )
        horizontal = (
            ((np.abs(y - box[1]) <= 7) | (np.abs(y - box[3]) <= 7))
            & (x > box[0] + trim)
            & (x < box[2] - trim)
        )
        region = region & (vertical | horizontal)
        inside = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
        fill_ref = (
            hex_rgb(unit["fill"])
            if unit.get("fill")
            else np.median(a[interior & source_support], axis=0)
        )
        fill_ref = np.median(a[interior & source_support], axis=0)
        fill_pred = (
            np.median(b[interior & observed_support], axis=0)
            if (interior & observed_support).any()
            else fill_ref
        )

        def mass(image, color, fill):
            backgrounds = np.where(inside[..., None], fill, np.array([1.0, 1.0, 1.0]))
            direction = color - backgrounds
            alpha = np.clip(
                np.sum((image - backgrounds) * direction, axis=2)
                / np.maximum(np.sum(direction * direction, axis=2), 1e-8),
                0,
                1,
            )
            selection = region & (source_support if image is a else observed_support)
            return float(alpha[selection].sum())

        ratio = mass(b, pred_color, fill_pred) / max(mass(a, ref_color, fill_ref), 1e-8)
        result.append(
            {
                "family": "stroke_width",
                "score": stroke_score(1, ratio, options.get("stroke_width_half_ratio", 1.25)),
                "visible_ink_width_ratio": float(ratio),
                "method": "antialias_coverage_mass_in_matched_boundary_band",
            }
        )
    return result


def ink_field(image, config, support=None):
    array = rgb(image)
    width = min(config["border_width_px"], max(1, min(array.shape[:2]) // 4))
    ring = np.ones(array.shape[:2], dtype=bool)
    ring[width:-width, width:-width] = False
    background = np.median(array[ring], axis=0)
    contrast = np.linalg.norm(array - background, axis=2)
    selected = np.ones(contrast.shape, bool) if support is None else np.asarray(support, bool)
    if selected.shape != contrast.shape:
        raise ValueError("Typography support must match its crop")
    if not selected.any():
        return np.zeros_like(contrast), None, background
    peak = float(np.quantile(contrast[selected], config["foreground_quantile"]))
    if peak < 1 / 255:
        return np.zeros_like(contrast), None, background
    foreground = np.median(array[(contrast >= peak) & selected], axis=0)
    scale = max(float(np.linalg.norm(foreground - background)), 1 / 255)
    return np.clip(contrast / scale, 0, 1) * selected, foreground, background


def glyph_evidence(
    reference,
    candidate,
    reference_box,
    candidate_box,
    options=None,
    reference_support=None,
    candidate_support=None,
):
    config = options
    aa, bb = np.asarray(reference_box, float), np.asarray(candidate_box, float)
    centers = [(v[:2] + v[2:]) / 2 for v in [aa, bb]]
    size = (
        (np.ceil(np.maximum(aa[2:] - aa[:2], bb[2:] - bb[:2])) + 2 * config["crop_padding_px"])
        .astype(int)
        .tolist()
    )
    crops = []
    support_crops = []
    for image, center, support in zip(
        [reference, candidate], centers, [reference_support, candidate_support]
    ):
        origin = np.rint(center - np.asarray(size) / 2).astype(int)
        box = [*origin, *(origin + size)]
        valid = [
            max(0, box[0]),
            max(0, box[1]),
            min(image.width, box[2]),
            min(image.height, box[3]),
        ]
        if valid[0] >= valid[2] or valid[1] >= valid[3]:
            return {
                "score": None,
                "status": "text_crop_outside_canvas",
                "algorithm": config["algorithm"],
            }
        crop = image.crop(valid).convert("RGB")
        pixels = np.asarray(crop)
        border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
        background = tuple(np.median(border, axis=0).round().astype(int))
        canvas = Image.new("RGB", tuple(size), background)
        canvas.paste(crop, (int(valid[0] - box[0]), int(valid[1] - box[1])))
        crops.append(canvas)
        selected = None
        if support is not None:
            support = np.asarray(support, bool)
            if support.shape != (image.height, image.width):
                raise ValueError("Typography support must match its image canvas")
            selected = np.zeros((size[1], size[0]), bool)
            x, y = int(valid[0] - box[0]), int(valid[1] - box[1])
            selected[y : y + crop.height, x : x + crop.width] = support[
                valid[1] : valid[3], valid[0] : valid[2]
            ]
        support_crops.append(selected)
    a, foreground_a, background_a = ink_field(crops[0], config, support_crops[0])
    b, foreground_b, background_b = ink_field(crops[1], config, support_crops[1])
    sigma = config["ink_blur_sigma_px"]
    if sigma:
        a, b = [cv2.GaussianBlur(x, (0, 0), sigma) for x in [a, b]]
    mass_a, mass_b = float(a.sum()), float(b.sum())
    if foreground_a is None or mass_a < 1:
        return {
            "score": None,
            "status": "reference_ink_unobservable",
            "algorithm": config["algorithm"],
        }
    if foreground_b is None or mass_b < 1:
        return {
            "score": 0.0,
            "status": "prediction_missing",
            "algorithm": config["algorithm"],
            "normalized_ink_overlap": 0.0,
            "ink_mass_score": 0.0,
            "candidate_to_reference_ink_mass": 0.0,
        }
    best = (-1.0, float("inf"), (0, 0))
    for dx in config["raster_alignment_offsets_px"]:
        for dy in config["raster_alignment_offsets_px"]:
            shifted = cv2.warpAffine(
                b.astype(np.float32),
                np.array([[1, 0, dx], [0, 1, dy]], np.float32),
                tuple(size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            # Keep the original denominator, including any ink moved outside
            # the observation window, so alignment cannot discard extra ink.
            agreement = float(np.minimum(a / mass_a, shifted / mass_b).sum())
            distance = abs(dx) + abs(dy)
            if agreement > best[0] + 1e-8 or (
                abs(agreement - best[0]) <= 1e-8 and distance < best[1]
            ):
                best = (agreement, distance, (dx, dy))
    delta_e = float(np.linalg.norm(srgb_to_lab(foreground_a) - srgb_to_lab(foreground_b)))
    appearance = half_score(
        max(0.0, delta_e - config["foreground_color_deadzone_deltaE"]),
        config["foreground_color_half_deltaE"],
    )
    mass_score = stroke_score(mass_a, mass_b, config["ink_mass_half_ratio"])
    return {
        "score": float((np.clip(best[0], 0, 1) * appearance * mass_score) ** (1 / 3)),
        "status": "measured",
        "algorithm": config["algorithm"],
        "normalized_ink_overlap": float(np.clip(best[0], 0, 1)),
        "ink_mass_score": mass_score,
        "foreground_color_score": appearance,
        "foreground_deltaE76": delta_e,
        "candidate_to_reference_ink_mass": mass_b / mass_a,
        "shared_alignment_px": list(best[2]),
        "reference_foreground_rgb": foreground_a.tolist(),
        "candidate_foreground_rgb": foreground_b.tolist(),
        "reference_background_rgb": background_a.tolist(),
        "candidate_background_rgb": background_b.tolist(),
    }
