"""Score one delivered slide from its image, source contract and independently collected observations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data import read_json, resolve_observations, sha256, support_mask
from evaluate import evaluate, leaf
from geometry import relation_feature, compare_geometry
from metrics import visual_total, macro_score


def bound(leaves, value):
    return macro_score([dict(r, score=value, state="imputed_bound")
                        if r["score"] is None else r for r in leaves])["score"]


def extent(mask):
    y,x=np.nonzero(mask)
    return int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)


def physical_score(reference,render,model,observed,config):
    if observed is None:return None
    if not observed.any():return 0.
    x0,y0,x1,y1=extent(model|observed)
    h,w=model.shape
    view=np.s_[max(0,y0-2):min(h,y1+2),max(0,x0-2):min(w,x1+2)]
    l,t,r,b=extent(model)
    return compare_geometry(reference[view].astype('float32')/255,render[view].astype('float32')/255,
        np.ones_like(model[view]),model[view],observed[view],math.hypot(r-l,b-t),(w,h),config)['score']


def terminal_label(labels,point):
    if point is None:return None
    if point=='missing':return 0
    # Port annotations are pixel coordinates with subpixel rounding permitted.
    x,y=np.rint(point).astype(int);h,w=labels.shape
    near=labels[max(y-2,0):min(y+3,h),max(x-2,0):min(x+3,w)]
    yi,xi=np.nonzero(near)
    if not len(xi):return 0
    xx=xi+max(x-2,0);yy=yi+max(y-2,0);r=(xx-point[0])**2+(yy-point[1])**2
    j=int(np.argmin(r));return int(labels[yy[j],xx[j]]) if r[j]<=4 else 0



def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_support(gt, key, size):
    support = support_mask(gt["visible_instance_supports"][key], size)
    require(support is not None and support.any(), f"Missing source ink for {key}")
    return support


def candidate_support(assignment, ledger, size):
    if assignment["state"] == "missing":
        return np.zeros(size[::-1], dtype=bool)
    if assignment["state"] == "matched":
        return support_mask(ledger["instances"][assignment["candidate_id"]], size)
    return None


def logical_ink(ink, crossings, occluder_mask):
    """A bridge adds connectivity only when its entire hidden span has an observed owner."""
    logical = ink.copy()
    for witness in crossings:
        a, b = tuple(witness["first"]), tuple(witness["second"])
        if not all(0 <= x < ink.shape[1] and 0 <= y < ink.shape[0] for x, y in [a, b]):
            raise ValueError("Crossing endpoint outside canvas")
        span = np.zeros(ink.shape, np.uint8)
        cv2.line(span, a, b, 1, 1)
        hidden = (span != 0) & ~ink
        owner = occluder_mask(witness["occluder"])
        require(ink[a[1], a[0]] and ink[b[1], b[0]] and hidden.any() and
                owner is not None and owner[hidden].all(), "Unsupported occlusion bridge")
        logical |= hidden
    return logical


def port_labels(ink, ports):
    h, w = ink.shape
    for xy in ports.values():
        if xy not in (None, "missing"):
            require(len(xy) == 2 and 0 <= xy[0] < w and 0 <= xy[1] < h, "Port outside canvas")
    _, labels = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    return {name: terminal_label(labels, xy) for name, xy in ports.items()}


def path_score(path, observed, components, assignments):
    start, end = path["source"], path["target"]
    states = [assignments.get(k, {}).get("state") for k in (start, end)]
    left, right = components.get(start), components.get(end)
    if "missing" in states or observed.get("connected") is False:
        return 0.
    if left is not None and right is not None and (not left or left != right):
        return 0.
    direction = observed.get("direction")
    if direction in ("forward", "reverse", "none") and direction != path["direction"]:
        return 0.
    if any(state != "matched" for state in states) or left is None or right is None or direction in (None, "unknown") or observed.get("connected") is None:
        return None
    require(direction in ("forward", "reverse", "none"), "Unrecognized observed direction")
    return float(direction == path["direction"])


def split_caps(ink, caps):
    occupied = np.zeros_like(ink)
    for cap in caps.values():
        if cap is None:
            continue
        require(not (cap & (~ink | occupied)).any(), "A physical tip must have disjoint ink owned by its network")
        occupied |= cap
    return None if any(v is None for v in caps.values()) else ink & ~occupied


def measure_geometry(gt, source, render, ledger, contract, observed, config):
    size = source.size
    source_rgb, candidate_rgb = np.asarray(source), np.asarray(render)
    matches = ledger["matches"]
    src_mask = lambda k: source_support(gt, k, size)
    def cand_mask(k):
        return candidate_support(matches.get(k, {"state": "unresolved"}), ledger, size)
    geometry, relations, extra_paths = [], [], []
    source_free = contract["free"]
    require(set(observed["free"]) <= set(source_free), "Undeclared free feature")
    for feature_id, spec in source_free.items():
        owner, family = spec["owner"], spec["family"]
        require(family in ("arrowhead", "curve"), "Unknown free geometry family")
        src, cand = src_mask(owner), cand_mask(owner)
        require(relation_feature(source, source, spec, family, config, src, src) > .99999,
                f"Source does not reproduce {feature_id}")
        opt = observed["free"].get(feature_id, "unknown")
        require(opt in ("on_parent", "missing", "unknown"), "Invalid free observation")
        value = None
        if matches.get(owner, {}).get("state") == "missing" or opt == "missing":
            value = 0.
        elif opt == "on_parent" and cand is not None:
            value = relation_feature(source, render, spec, family, config, cand, src) if cand.any() else 0.
        geometry.append(leaf(owner, family, value, id=feature_id))
    require(set(observed["networks"]) <= set(contract["networks"]), "Undeclared network")
    for key, network in contract["networks"].items():
        candidate = observed["networks"].get(key, {})
        src, cand = src_mask(key), cand_mask(key)
        require(all(p["direction"] in ("forward", "none") for p in network["paths"]), "Use ordered ends for a directional path")
        expected = {(p["source"], p["target"]): p for p in network["paths"]}
        require(expected and len(expected) == len(network["paths"]), "Duplicate or empty source path list")
        src_caps = {k: support_mask(h, size) for k, h in network["heads"].items()}
        require(all(m is not None and m.any() for m in src_caps.values()), "Missing source cap ink")
        src_shaft = split_caps(src, src_caps)
        require(src_shaft.any(), "Empty source shaft")
        source_labels = port_labels(logical_ink(src, network.get("occlusions", []), src_mask), network["ports"])
        for a, b in expected:
            require(source_labels.get(a) and source_labels[a] == source_labels.get(b),
                    f"Source path lacks ink or a witnessed crossing: {a}, {b}")
        labels = {}
        if cand is not None:
            labels = port_labels(logical_ink(cand, candidate.get("occlusions", []), cand_mask), candidate.get("ports", {}))
        readings = {(v["source"], v["target"]): v for v in candidate.get("paths", [])}
        require(len(readings) == len(candidate.get("paths", [])), "Duplicate observed path")
        for (a, b), reading in readings.items():
            require(reading.get("connected") is None or isinstance(reading["connected"], bool), "Invalid continuity flag")
            require(reading.get("direction") in ("forward", "reverse", "none", "unknown", None), "Invalid path direction")
            directed = (b, a) if reading.get("direction") == "reverse" else (a, b)
            if reading.get("connected") and directed not in expected and not (
                (b, a) in expected and expected[b, a]["direction"] == reading.get("direction") == "none"
            ):
                extra_paths.append(dict(network=key, **reading))
        absent = matches.get(key, {}).get("state") == "missing"
        for ends, path in expected.items():
            reading = readings.get(ends)
            if reading is None:
                reverse = readings.get(ends[::-1])
                if reverse is not None:
                    direction = reverse.get("direction")
                    reading = {**reverse, "direction": {"forward": "reverse", "reverse": "forward"}.get(direction, direction)}
                else:
                    reading = {"connected": False} if candidate.get("paths_complete") is True else {}
            value = 0. if absent else path_score(path, reading, labels, matches)
            relations.append(leaf(key, "relation", value, source=ends[0], target=ends[1]))
        heads = candidate.get("heads", {})
        require(set(heads) <= set(src_caps), "Undeclared physical head")
        caps = {}
        for head_id in src_caps:
            entry = heads.get(head_id)
            caps[head_id] = np.zeros(size[::-1], bool) if absent or entry == "missing" else (
                support_mask(entry, size) if isinstance(entry, dict) and cand is not None else None)
        shaft = split_caps(cand, caps) if cand is not None else None
        geometry.append(leaf(key, "curve", physical_score(source_rgb, candidate_rgb, src_shaft, shaft, config), id="shaft"))
        geometry.extend(leaf(key, "arrowhead", physical_score(source_rgb, candidate_rgb, sm, caps[k], config), id=k)
                        for k, sm in src_caps.items())
    return geometry, relations, extra_paths


def score(case_file):
    lock_file = Path(__file__).with_name("lock.json")
    lock = read_json(lock_file)
    for path, digest in lock["implementation"].items():
        require(sha256(ROOT / path) == digest, f"RC1 implementation changed: {path}")
    case = read_json(case_file)
    paths = {k: case_file.parent / case[k] for k in ["image", "gt", "pptx", "rendered", "observations"]}
    files = {k: sha256(p) for k, p in paths.items()}
    gt, judgments = read_json(paths["gt"]), read_json(paths["observations"])
    require(judgments["artifact_sha256"] == files["pptx"] and judgments["source_gt_sha256"] == files["gt"],
            "Observations belong to a different submission or source GT")
    source, render = [Image.open(paths[k]).convert("RGB") for k in ("image", "rendered")]
    config = read_json(ROOT / "metric.json")
    contract_digest = hashlib.sha256(json.dumps(gt["visual_contract"], sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
    require(contract_digest == gt["visual_contract_sha256"] and
            gt["visual_contract"]["source_image_sha256"] == files["image"], "Source contract mismatch")
    regular = evaluate(gt, source, render, judgments, config)
    components = regular["components"]
    present = {(branch, r["unit"], r["family"]) for branch, leaves in components.items() for r in leaves}
    present |= {("L_relations", r["unit"], "relation") for r in regular["relations"]}
    unbudgeted = [r for r in regular["missing"] if tuple(r) not in present and r[0] != "H"]
    source_complete = bool(gt.get("complete") or gt.get("visual_coverage_complete"))
    declaration = gt.get("required_supplement")
    require(not gt.get("required_network_budget"), "Normalize the network budget to rc1 first")
    extension_g, extension_l, extra_paths = [], [], []
    if declaration:
        descriptor = case_file.parent / case["geometry"]
        files["geometry"] = sha256(descriptor)
        require(files["geometry"] == declaration["sha256"] == gt["visual_contract"]["pending_source_supplement_sha256"],
                "Source geometry digest mismatch")
        contract = read_json(descriptor)
        require(contract["schema"] == "eb-f-geometry-rc1" and contract["image_sha256"] == files["image"],
                "Wrong source geometry contract")
        source_complete = contract["complete"] is True
        observed_file = case_file.parent / case["geometry_observations"]
        files["geometry_observations"] = sha256(observed_file)
        observation = read_json(observed_file)
        require(observation["bindings"] == {k: files[k] for k in ("geometry", "observations", "rendered", "pptx")},
                "Geometry observations belong to a different submission")
        extension_g, extension_l, extra_paths = measure_geometry(gt, source, render, resolve_observations(gt, source, render, judgments),
                                                   contract, observation, config)
    else:
        require("geometry" not in case and "geometry_observations" not in case and
                not gt["visual_contract"].get("pending_source_supplement_sha256"), "Unbound auxiliary geometry")
    geometry = components["G"] + extension_g
    relations = regular["relations"] + extension_l
    H = regular["H"]
    bounds = {}
    for endpoint in (0., 1.):
        layout = bound(components["L_layout"], endpoint)
        if relations and layout is not None:
            share = config["layout"]["layout_weight"]
            layout = share * layout + (1-share)*mean(endpoint if r["score"] is None else r["score"] for r in relations)
        parts = dict(L=layout, G=bound(geometry, endpoint), A=bound(components["A"], endpoint), T=bound(components["T"], endpoint))
        f = None
        if source_complete and not unbudgeted and all(v is not None for v in parts.values()):
            f = visual_total(parts, H if H is not None else 0, config["branch_weights"])
            if H is None and endpoint == 0:
                f = 0.
        for branch, val in {**parts, "F": f}.items():
            bounds.setdefault(branch, []).append(val)
    rgb = np.asarray(render, float)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:,0], rgb[:,-1]])
    if source_complete and np.max(abs(rgb-np.median(border, axis=0))) <= 12:
        bounds["F"] = [0., 0.]
    all_leaves = [r for v in components.values() for r in v] + regular["relations"] + extension_g + extension_l
    lo, hi = bounds["F"]
    return dict(protocol="eb-f-single-rc1", lock_sha256=sha256(lock_file), inputs=files, config_sha256=sha256(ROOT/"metric.json"),
                F=lo if lo is not None and abs(hi-lo)<1e-9 else None, bounds=bounds, H=H,
                eligible=len(all_leaves), measured=sum(r["score"] is not None for r in all_leaves),
                unbudgeted=unbudgeted, components=components, relations=regular["relations"],
                geometry_extension=extension_g, relation_extension=extension_l, extra_observed_paths=extra_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = score(args.case.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("F", "bounds", "H", "eligible", "measured")}, allow_nan=False))
