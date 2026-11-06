"""Evaluate normalized image/PPT pairs. Run: uv run evaluate.py experiments/current.jsonl"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data import (
    assigned_instance,
    automatic_correspondence,
    missing_requirements,
    read_json,
    resolve_observations,
    rubric_score,
    sha256,
    support_mask,
)
from geometry import feature_score, relation_feature, rgb, unit_features
from metrics import (
    color_score,
    content_score,
    glyph_evidence,
    hex_rgb,
    layout_score,
    macro_score,
    stroke_score,
    unit_appearance,
    visual_total,
)
from render import validate_pptx


def relation_score(ledger, relation):
    assignment, instance = assigned_instance(ledger, relation["id"])
    if assignment["state"] != "matched":
        return 0.0 if assignment["state"] == "missing" else None
    a, b = [assigned_instance(ledger, relation[k])[0] for k in ("source", "target")]
    if "missing" in (a["state"], b["state"]) or instance.get("continuous") is False:
        return 0.0
    if "unresolved" in (a["state"], b["state"]):
        return None
    if any(instance.get(k) is None for k in ("endpoints", "direction")):
        return None
    endpoints = instance["endpoints"]
    expected = [a["candidate_id"], b["candidate_id"]]
    direction = relation.get("direction", "forward" if relation.get("directed", True) else "none")

    def arcs(nodes, direction):
        assert direction in ("forward", "reverse", "both", "none")
        return ({tuple(nodes)} if direction in ("forward", "both") else set()) | (
            {tuple(reversed(nodes))} if direction in ("reverse", "both") else set()
        )

    if direction == "none":
        correct = instance["direction"] == "none" and set(endpoints) == set(expected)
    else:
        correct = arcs(endpoints, instance["direction"]) == arcs(expected, direction)
    if not correct:
        return 0.0
    return 1.0 if instance.get("continuous") is True else None


def leaf(unit, family, score, region="page", **details):
    return {
        "unit": unit,
        "region": region,
        "family": family,
        "score": score,
        "state": "unknown" if score is None else "measured",
        **details,
    }


def evaluate(gt, reference, candidate, judgments, config):
    """Compute F from pixels and external observations; no PPT object-tree assumptions."""
    ledger = resolve_observations(gt, reference, candidate, judgments)
    source_masks = {
        rid: support_mask(value, reference.size)
        for rid, value in gt["visible_instance_supports"].items()
    }
    candidate_masks = {
        cid: support_mask(value, reference.size) for cid, value in ledger["instances"].items()
    }
    layout, geometry, appearance, text, relations = [], [], [], [], []
    options = config["layout"]
    for unit in gt["units"]:
        rid = unit["id"]
        assignment, instance = assigned_instance(ledger, rid)
        state = assignment["state"]
        box = instance["bbox"] if state == "matched" else None
        match = {
            "bbox": box,
            "status": {
                "matched": "matched",
                "missing": "prediction_missing",
                "unresolved": "measurement_unavailable",
            }[state],
        }
        source = source_masks.get(rid)
        observed = candidate_masks.get(assignment.get("candidate_id"))
        region = unit.get("region", "page")
        if unit["kind"] != "text":
            value = (
                None
                if state == "unresolved"
                else layout_score(
                    unit["bbox"], box, options["position_half_error"], options["size_half_ratio"]
                )
            )
            layout.append(leaf(rid, "layout", value, region))
        for feature in unit_features(unit, reference.size, config["geometry"]):
            evidence = feature_score(
                reference, candidate, unit, match, feature, config, observed, source
            )
            geometry.append(
                leaf(
                    rid,
                    feature["family"],
                    evidence["score"],
                    region,
                    id=feature["id"],
                    evidence=evidence,
                )
            )
        for value in unit_appearance(reference, candidate, unit, match, config, observed, source):
            score = value["score"]
            if state == "matched" and (source is None or observed is None):
                score = None
            appearance.append(leaf(rid, value["family"], score, region))

    for relation in gt["relations"]:
        rid = relation["id"]
        assignment, instance = assigned_instance(ledger, rid)
        state = assignment["state"]
        source = source_masks.get(rid)
        observed = candidate_masks.get(assignment.get("candidate_id"))
        relations.append(leaf(rid, "relation", relation_score(ledger, relation)))
        for family in ("arrowhead", "curve"):
            if family == "arrowhead" and not relation.get("directed", True):
                continue
            value = 0.0 if state == "missing" else None
            if state == "matched":
                value = relation_feature(
                    reference, candidate, relation, family, config, observed, source
                )
            geometry.append(leaf(rid, family, value))
        for family, key in (("stroke", "stroke_rgb"), ("stroke_width", "stroke_width_px")):
            if relation.get(family) is None:
                continue
            value = 0.0 if state == "missing" else None
            observation = instance.get(key) if instance else None
            if state == "matched" and observation is not None:
                value = (
                    color_score(
                        hex_rgb(relation[family]),
                        observation,
                        config["appearance"]["deltaE76_half_error"],
                    )
                    if family == "stroke"
                    else stroke_score(
                        relation[family],
                        observation,
                        config["appearance"]["stroke_width_half_ratio"],
                    )
                )
            appearance.append(leaf(rid, family, value))

    for target in gt["fullpage_text"]["text_regions"]:
        rid = target["id"]
        assignment, instance = assigned_instance(ledger, rid)
        state = assignment["state"]
        source_box = ledger["reference_slots"][rid].get("bbox", target["bbox"])
        content = glyph = position = 0.0 if state == "missing" else None
        if target.get("verified") is False:
            content = None
        evidence = {}
        if state == "matched":
            position = layout_score(source_box, instance["bbox"])
            observed_text = instance.get("text")
            if observed_text is not None and target.get("verified") is not False:
                expected_text = target["text"]
                if target.get("whitespace_semantic") is False:
                    expected_text, observed_text = (
                        "".join(expected_text.split()),
                        "".join(observed_text.split()),
                    )
                content = content_score(expected_text, observed_text)
            source = source_masks.get(rid)
            observed = candidate_masks.get(assignment["candidate_id"])
            if source is not None and observed is not None:
                evidence = glyph_evidence(
                    reference,
                    candidate,
                    source_box,
                    instance["bbox"],
                    config["typography"],
                    source,
                    observed,
                )
                glyph = evidence["score"]
        region = target.get("region", "page")
        layout.append(leaf(rid, "layout", position, region))
        text += [
            leaf(rid, "content", content, region),
            leaf(rid, "typography", glyph, region, evidence=evidence),
        ]

    if gt.get("background"):
        pixels = rgb(candidate)
        border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
        appearance.append(
            leaf(
                "page",
                "background",
                color_score(hex_rgb(gt["background"]), np.median(border, axis=0)),
            )
        )
    for requirement in gt["rubric_requirements"]:
        target = {"A": appearance, "T": text}[requirement["branch"]]
        rid, family = requirement["unit"], requirement["family"]
        if requirement.get("replace"):
            target[:] = [x for x in target if (x["unit"], x["family"]) != (rid, family)]
        target.append(
            leaf(
                rid,
                family,
                rubric_score(requirement, judgments, ledger),
                requirement.get("region", "page"),
                requirement_id=requirement["id"],
            )
        )

    components = {"L_layout": layout, "G": geometry, "A": appearance, "T": text}
    aggregates = {key: macro_score(leaves) for key, leaves in components.items()}
    layout_value = aggregates["L_layout"]["score"]
    relation_value = (
        float(np.mean([r["score"] for r in relations]))
        if relations and all(r["score"] is not None for r in relations)
        else None
    )
    complete = bool(gt.get("complete") or gt.get("visual_coverage_complete"))
    L = None
    if complete:
        if not relations:
            L = layout_value
        elif layout_value is not None and relation_value is not None:
            L = (
                options["layout_weight"] * layout_value
                + (1 - options["layout_weight"]) * relation_value
            )
    scores = {"L": L, **{b: aggregates[b]["score"] for b in ("G", "A", "T")}}
    missing = missing_requirements(gt, components, relations, ledger["H"])
    F = (
        visual_total(scores, ledger["H"], config["branch_weights"])
        if complete and not missing
        else None
    )
    pixels = rgb(candidate)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    blank = not (np.max(np.abs(pixels - np.median(border, axis=0)), axis=2) > 12 / 255).any()
    if complete and blank:
        F = 0.0
    return {
        "F": F,
        **scores,
        "H": ledger["H"],
        "families": aggregates,
        "components": components,
        "relations": relations,
        "missing": missing,
        "extra_instances": ledger["extras"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("metric.json"))
    parser.add_argument("--output", type=Path, default=Path("results/current"))
    parser.add_argument("--id", nargs="+", help="Select sample IDs")
    parser.add_argument("--match-only", action="store_true", help="Automatically propose correspondence from image, GT and PPTX; no preassigned observations are read")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render, save and reopen PPTX instead of using frozen PNGs",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.id:
        rows = [r for r in rows if r["id"] in args.id]
    assert rows, "No samples selected"
    args.output.mkdir(parents=True, exist_ok=True)
    table = []
    for row in rows:
        start = time.monotonic()
        print(f"Scoring {row['id']} ...", flush=True)
        paths = {
            key: (args.manifest.parent / row[key]).resolve()
            for key in ("image", "gt", "pptx")
        }
        gt = read_json(paths["gt"])
        reference = Image.open(paths["image"]).convert("RGB")
        validity = None
        if args.render or not row.get("rendered"):
            validity = validate_pptx(paths["pptx"], args.output / row["id"], reference.size)
            png = validity["rendered"]
        else:
            png = args.manifest.parent / row["rendered"]
        if args.match_only:
            result = automatic_correspondence(gt, reference, Image.open(png).convert("RGB"), paths["pptx"])
            result.update(id=row["id"], seconds=time.monotonic()-start,
                          inputs={key:sha256(path) for key,path in paths.items()},
                          candidate_png_sha256=sha256(png))
            (args.output / (row['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            counts = {state:sum(m['state']==state for m in result['assignments'].values()) for state in ('matched','unresolved')}
            table.append({'id':row['id'],**counts,'fragment_groups':sum(x['origin']=='fragment_union' for x in result['instances'].values()),'seconds':result['seconds']})
            print(f"{row['id']}: automatic {counts}, {result['seconds']:.1f}s",flush=True)
            continue
        paths['observations'] = (args.manifest.parent / row['observations']).resolve()
        judgments = read_json(paths['observations'])
        assert sha256(paths['pptx']) == judgments['artifact_sha256'], 'Observations belong to another PPTX'
        result = evaluate(gt, reference, Image.open(png).convert("RGB"), judgments, config)
        result.update(
            id=row["id"],
            V=validity,
            E=None,
            D=None,
            seconds=time.monotonic() - start,
            config_sha256=sha256(args.config),
            inputs={key: sha256(path) for key, path in paths.items()},
        )
        (args.output / (row["id"] + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        table.append({k: result[k] for k in ("id", "F", "L", "G", "A", "T", "H")})
        print(f"{row['id']}: F={result['F']}, {result['seconds']:.1f}s", flush=True)
    with (args.output / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
