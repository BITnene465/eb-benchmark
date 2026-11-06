"""Read normalized source annotations and artifact-bound visual observations."""

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import math
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(image):
    image = image.convert("RGB")
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def support_mask(instance, size):
    polygons = instance.get("support_polygons")
    if polygons is None:
        return None
    mask = Image.new("1", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon([tuple(p) for p in polygon], fill=1)
    return np.asarray(mask, bool).copy()


def assigned_instance(ledger, reference_id):
    match = ledger["matches"].get(reference_id, {"state": "unresolved"})
    return match, ledger["instances"].get(match.get("candidate_id"))


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    return inter / max(float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - inter, 1e-9)


def extra_evidence_budget(reference_fields, extra_fields):
    """Candidate H: union visible evidence per class, then source-normalized mass.

    Inputs must be observed nonnegative evidence fields on a common raster,
    not object boxes or object counts. Max-union prevents duplicated proposals
    from spending the same visible evidence twice. Ownership is an input to
    validate, not something this arithmetic function infers.
    """
    def mass(fields):
        if not fields:
            return 0.0
        arrays = [np.asarray(x, float) for x in fields]
        if any(not np.isfinite(x).all() or (x < 0).any() for x in arrays):
            raise ValueError('Evidence must be finite and nonnegative')
        return float(np.maximum.reduce(arrays).sum())

    sources = {kind: mass(fields) for kind, fields in reference_fields.items()}
    extras = {kind: mass(fields) for kind, fields in extra_fields.items()}
    classes = [kind for kind, value in sources.items() if value > 0]
    uncalibrated = [kind for kind, value in extras.items() if value > 0 and sources.get(kind, 0) == 0]
    if not classes or uncalibrated:
        return {'H': None, 'uncalibrated_classes': uncalibrated}
    ratios = {kind: extras.get(kind, 0) / sources[kind] for kind in classes}
    return {'H': sum(ratios.values()) / len(classes), 'class_ratios': ratios,
            'reference_mass': sources, 'extra_mass': extras,
            'scope': 'candidate visible-evidence budget; requires validated ownership'}


def ppt_proposals(pptx, size):
    """Content proposals; object names and source/candidate annotation IDs are not used."""
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    with zipfile.ZipFile(pptx) as z:
        page = ET.fromstring(z.read("ppt/presentation.xml")).find("p:sldSz", ns)
        tree = ET.fromstring(z.read("ppt/slides/slide1.xml")).find("p:cSld/p:spTree", ns)
    root = np.diag([size[0] / int(page.get("cx")), size[1] / int(page.get("cy")), 1.0])
    proposals = []

    def has_paint(properties):
        # Missing paint may inherit a theme; explicit noFill/zero alpha cannot.
        if properties is None:
            return True
        if properties.find('a:noFill', ns) is not None:
            return False
        paint_nodes = [c for c in properties if c.tag != '{'+ns['a']+'}ln']
        colors = [x for node in paint_nodes for x in node.iter() if x.tag.rsplit('}', 1)[-1] in
                  ('srgbClr', 'schemeClr', 'scrgbClr', 'sysClr', 'prstClr')]
        return not colors or any(
            c.find('a:alpha', ns) is None or int(c.find('a:alpha', ns).get('val')) > 0
            for c in colors)

    def transform(x, group=False):
        off, ext = x.find("a:off", ns), x.find("a:ext", ns)
        ox, oy = float(off.get("x")), float(off.get("y"))
        w, h = float(ext.get("cx")), float(ext.get("cy"))
        angle = math.radians(float(x.get("rot", 0)) / 60000)
        c, s = math.cos(angle), math.sin(angle)
        fx, fy = (-1 if x.get("flipH") in ("1", "true") else 1), (-1 if x.get("flipV") in ("1", "true") else 1)
        rotation = np.array([[c*fx, -s*fy, 0], [s*fx, c*fy, 0], [0, 0, 1.]])
        rotation[:2, 2] = np.array([ox+w/2, oy+h/2]) - rotation[:2, :2] @ np.array([w/2, h/2])
        if not group:
            return rotation, w, h
        child, extent = x.find("a:chOff", ns), x.find("a:chExt", ns)
        scale = np.array([[w/max(float(extent.get('cx')), 1), 0, 0], [0, h/max(float(extent.get('cy')), 1), 0], [0, 0, 1.]])
        scale[:2, 2] = -scale[:2, :2] @ np.array([float(child.get('x')), float(child.get('y'))])
        return rotation @ scale, w, h

    def walk(parent, matrix):
        for element in parent:
            kind = element.tag.rsplit("}", 1)[-1]
            if kind not in ("grpSp", "sp", "pic", "cxnSp", "graphicFrame"):
                continue
            identity = element.find('.//p:cNvPr', ns)
            if identity is not None and identity.get('hidden') in ('1', 'true'):
                continue
            x = element.find("p:grpSpPr/a:xfrm" if kind == "grpSp" else "p:spPr/a:xfrm", ns)
            if x is None:
                continue
            local, w, h = transform(x, kind == "grpSp")
            if kind == "grpSp":
                walk(element, matrix @ local)
                continue
            corners = (matrix @ local @ np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]]))[:2]
            lo, hi = np.maximum(corners.min(1), 0), np.minimum(corners.max(1), size)
            # Lines have zero OOXML extent along one axis, but visible raster width.
            hi = np.minimum(np.maximum(hi, lo + 2), size)
            if np.any(hi <= lo):
                continue
            box = [*lo.tolist(), *hi.tolist()]
            value = " ".join("".join(p.itertext()).strip() for p in element.findall(".//a:t", ns)).strip()
            properties = element.find('p:spPr', ns)
            geometry = element.find("p:spPr/a:prstGeom", ns)
            shape = geometry.get("prst", "") if geometry is not None else ""
            paths = element.findall('p:spPr/a:custGeom/a:pathLst/a:path', ns)
            ends = element.findall('p:spPr/a:ln/a:headEnd', ns) + element.findall('p:spPr/a:ln/a:tailEnd', ns)
            open_arrow = (bool(paths) and all(p.find('a:close', ns) is None for p in paths)
                          and (properties.find('a:noFill', ns) is not None or all(p.get('fill') == 'none' for p in paths))
                          and any(end.get('type', 'none') != 'none' for end in ends))
            connection = kind == "cxnSp" or open_arrow or any(token in shape.lower() for token in ("arrow", "line", "connector"))
            classes = ["graphic", "shape", "connection"] if kind == "pic" else (["connection"] if connection else ["shape", "graphic"])
            if value:
                proposals.append({"id": f"p{len(proposals)}", "bbox": box, "classes": ["text"], "text": value, "origin": "native_text"})
            explicit_fill = any(properties.find('a:'+k, ns) is not None
                                for k in ('solidFill', 'gradFill', 'blipFill', 'pattFill'))
            filled = has_paint(properties) and kind != 'cxnSp' and (not value or explicit_fill)
            line = element.find('p:spPr/a:ln', ns)
            outlined = (line is not None and has_paint(line)) or (
                line is None and element.find('p:style/a:lnRef', ns) is not None)
            if kind == 'pic' or filled or outlined:
                proposals.append({"id": f"p{len(proposals)}", "bbox": box, "classes": classes, "origin": "native_geometry" if kind != "pic" else "native_picture"})
    walk(tree, root)
    return proposals


def candidate_ocr(candidate):
    """Run the installed macOS OCR on the candidate only; cache by pixel digest."""
    root=Path(__file__).parent
    cache=root/'.runtime'/'candidate-ocr';cache.mkdir(parents=True,exist_ok=True)
    source=root/'experiments'/'ocr_candidates.swift'
    binary=cache/('ocr-'+sha256(source)[:12])
    if not binary.exists():
        subprocess.run(['swiftc','-O',str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    stem=cache/pixel_digest(candidate)
    output=stem.with_suffix('.json')
    if not output.exists():
        candidate.save(stem.with_suffix('.png'))
        subprocess.run([str(binary),str(stem.with_suffix('.png')),str(output)],check=True,capture_output=True,text=True)
        stem.with_suffix('.png').unlink()
    return read_json(output)['regions']


def automatic_correspondence(gt, reference, candidate, pptx, ocr_regions=None):
    """Automatic proposal/assignment prototype; never reads preassigned observations.

    A proposal is not a validated visible segmentation. Returned leftovers and
    null assignments are review candidates, not confirmed extras/missing objects.
    """
    from difflib import SequenceMatcher
    size = reference.size
    slots = gt["reference_entities"]["instances"]
    text_by_id = {x["id"]: x["text"] for x in gt["fullpage_text"]["text_regions"]}
    text_verified = {x['id']: x.get('verified', True) for x in gt['fullpage_text']['text_regions']}
    primitives = ppt_proposals(pptx, size)
    def visible(proposal):
        x,y,u,v=proposal['bbox']
        pad=max(2,.005*max(u-x,v-y))
        window=(max(0,math.floor(x-pad)),max(0,math.floor(y-pad)),min(size[0],math.ceil(u+pad)),min(size[1],math.ceil(v+pad)))
        pixels=np.asarray(candidate.crop(window).convert('RGB'))
        return float(np.ptp(pixels.astype(float),axis=(0,1)).max())>=2
    primitives=[q for q in primitives if visible(q)]
    ocr_regions=candidate_ocr(candidate) if ocr_regions is None else ocr_regions
    characters=[]
    for region in ocr_regions:
        for i,char in enumerate(region['chars']):
            characters.append({'id':region['id']+f'-{i}','bbox':char['bbox'],'text':char['text'],'classes':['text'],'origin':'rendered_ocr'})
    primitives += characters
    for region in ocr_regions:
        primitives.append({'id':region['id'],'bbox':region['bbox'],'text':region['text'],'classes':['text'],'origin':'rendered_ocr','parts':[region['id']+f'-{i}' for i in range(len(region['chars']))]})
    pool = list(primitives)
    factor = min(1.0, 1280 / max(size))
    raster_size = tuple(max(1, round(v * factor)) for v in size)
    a = np.asarray(reference.convert("L").resize(raster_size))
    b = np.asarray(candidate.convert("L").resize(raster_size))

    # Image proposals preserve a route for rasterized or merged native objects.
    # The search is local and source-guided, with three globally fixed scales.
    generated=[]
    for slot in slots.values():
        box = np.asarray(slot["bbox"], float)
        x0,y0,x1,y1 = np.rint(box * factor).astype(int)
        patch = a[max(0,y0):min(a.shape[0],y1),max(0,x0):min(a.shape[1],x1)]
        if min(patch.shape, default=0) < 2 or patch.std() < 1:
            continue
        margin = max(24, round(np.linalg.norm(box[2:]-box[:2]) * factor * .65))
        sx,sy = max(0,x0-margin),max(0,y0-margin)
        ex,ey = min(b.shape[1],x1+margin),min(b.shape[0],y1+margin)
        search = b[sy:ey,sx:ex]
        proposals = []
        for scale in (.9,1.,1.1):
            w,h = max(2,round(patch.shape[1]*scale)),max(2,round(patch.shape[0]*scale))
            if w > search.shape[1] or h > search.shape[0]:
                continue
            correlation = cv2.matchTemplate(search,cv2.resize(patch,(w,h)),cv2.TM_CCOEFF_NORMED)
            _,score,_,(x,y) = cv2.minMaxLoc(correlation)
            if score >= .20:
                proposals.append((score,[(sx+x)/factor,(sy+y)/factor,(sx+x+w)/factor,(sy+y+h)/factor]))
        for score, box in sorted(proposals,reverse=True)[:2]:
            proposal={"bbox":box,"classes":[slot['class']],"origin":"raster_template"}
            if slot['class']=='text':
                inside=[c for c in characters if box[0]<=(c['bbox'][0]+c['bbox'][2])/2<=box[2] and box[1]<=(c['bbox'][1]+c['bbox'][3])/2<=box[3]]
                if not inside and score < .45:
                    continue
                if inside:
                    proposal.update(text=''.join(c['text'] for c in inside),parts=[c['id'] for c in inside])
                # Strong raster text can survive failed OCR. Its proposal ID
                # still shares the ordinary occupancy/overlap constraints.
            generated.append(proposal)
    # Geometry is shared; source-conditioned similarity is never shared.
    unique={(tuple(q['classes']),tuple(q['bbox'])):q for q in generated}
    for key in sorted(unique):
        q=unique[key]
        pool.append(dict(q,id=f'v{len(pool)}'))

    def union(items):
        boxes = np.array([x['bbox'] for x in items])
        return np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)].tolist()

    pair_cache = {}
    def pair_ncc(rid, slot, proposal):
        key=(rid,tuple(proposal['bbox']))
        if key not in pair_cache:
            ra=np.rint(np.asarray(slot['bbox'])*factor).astype(int)
            rb=np.rint(np.asarray(proposal['bbox'])*factor).astype(int)
            source_patch=a[max(0,ra[1]):min(a.shape[0],ra[3]),max(0,ra[0]):min(a.shape[1],ra[2])]
            target_patch=b[max(0,rb[1]):min(b.shape[0],rb[3]),max(0,rb[0]):min(b.shape[1],rb[2])]
            if min(source_patch.shape,default=0)<2 or min(target_patch.shape,default=0)<2 or source_patch.std()<1:
                score=-1.
            else:
                template=cv2.resize(source_patch,(target_patch.shape[1],target_patch.shape[0]))
                score=float(cv2.matchTemplate(target_patch,template,cv2.TM_CCOEFF_NORMED)[0,0])
            pair_cache[key]=score
        return pair_cache[key]

    def cost(rid, slot, proposal):
        ncc=pair_ncc(rid,slot,proposal) if proposal.get('origin')=='raster_template' else None
        if ncc is not None and (ncc < .20 or (slot['class']=='text' and not proposal.get('parts') and ncc < .45)):
            return float('inf')
        ra, rb = np.array(slot['bbox']),np.array(proposal['bbox'])
        sa,sb = np.maximum(ra[2:]-ra[:2],1),np.maximum(rb[2:]-rb[:2],1)
        distance = min(1.,np.linalg.norm((ra[:2]+ra[2:]-rb[:2]-rb[2:])/2)/(np.linalg.norm(sa)+.01*max(size)))
        scale = min(1.,float(np.abs(np.log(sb/sa)).mean())/math.log(3))
        ca = np.asarray(reference.crop(ra.tolist()).convert('L').resize((32,32)),float)
        cb = np.asarray(candidate.crop(rb.tolist()).convert('L').resize((32,32)),float)
        va,vb = ca-ca.mean(),cb-cb.mean()
        similarity = float((va*vb).sum()/max(np.linalg.norm(va)*np.linalg.norm(vb),1e-8))
        visual = (1-np.clip(similarity,-1,1))/2
        # A larger window checks the same object's surroundings. It prevents a
        # small glyph or line from relocating onto a similar neighbouring mark.
        context=[]
        for image,box0,extent in [(reference,ra,sa),(candidate,rb,sb)]:
            pad=np.maximum(extent*.6,8)
            context.append(np.asarray(image.crop(np.r_[box0[:2]-pad,box0[2:]+pad].tolist()).convert('L').resize((40,40)),float))
        x,y=[v-v.mean() for v in context]
        context_error=(1-np.clip(float((x*y).sum()/max(np.linalg.norm(x)*np.linalg.norm(y),1e-8)),-1,1))/2
        value = .40*distance+.20*scale+.20*visual+.20*context_error
        if slot['class']=='text' and proposal.get('text') and text_verified.get(rid, True):
            expected = ''.join(text_by_id.get(rid,'').split())
            observed = ''.join(proposal['text'].split())
            value = .7*value + .3*(1-SequenceMatcher(None,expected,observed).ratio())
        if proposal.get('origin')=='raster_template' and ncc<.45:
            value += .25
        return float(value)

    hypotheses = []
    centers=np.array([[(q['bbox'][0]+q['bbox'][2])/2,(q['bbox'][1]+q['bbox'][3])/2] for q in pool])
    spatial=cKDTree(centers) if len(centers) else None
    for rid,slot in sorted(slots.items()):
        rbox=np.asarray(slot['bbox']);center=(rbox[:2]+rbox[2:])/2
        nearby=spatial.query_ball_point(center,2*np.linalg.norm(rbox[2:]-rbox[:2])+.03*max(size)) if spatial else []
        eligible = [pool[i] for i in nearby if slot['class'] in pool[i]['classes']]
        ranked = sorted([(cost(rid,slot,q),q) for q in eligible],key=lambda x:x[0])[:6]
        proposals = [dict(q,members=q.get('parts') or [q['id']]) for _,q in ranked]
        native = [q for _,q in ranked if q['origin']!='raster_template']
        for count in (2,3):
            for group in combinations(native,count):
                # A group combines fragments; nested/duplicate whole-object boxes
                # must not be rewarded as extra members of the same object.
                if any(box_iou(x['bbox'],y['bbox'])>.65 for x,y in combinations(group,2)):
                    continue
                proposals.append({'bbox':union(group),'members':list(dict.fromkeys(m for q in group for m in (q.get('parts') or [q['id']]))),
                                  'origin':'fragment_union','text':' '.join(q.get('text','') for q in sorted(group,key=lambda v:(v['bbox'][1],v['bbox'][0])))})
        for q in proposals:
            c = cost(rid,slot,q)
            if c < .32:
                if q.get('origin')=='raster_template':
                    q=dict(q,pair_ncc=pair_ncc(rid,slot,q))
                hypotheses.append({'reference_id':rid,'class':slot['class'],'cost':c,**q})
    if not hypotheses:
        return {'assignments':{rid:{'state':'unresolved'} for rid in slots},'instances':{},'inventory_complete':False,'unclaimed_primitives':primitives,'diagnostics':{'hypotheses':0}}

    resources = {('source',rid):i for i,rid in enumerate(sorted(slots))}
    for q in pool:
        resources[('primitive',q['id'])] = len(resources)
    constraint = lil_matrix((len(resources),len(hypotheses)),dtype=float)
    for j,h in enumerate(hypotheses):
        constraint[resources[('source',h['reference_id'])],j] = 1
        for member in h['members']:
            constraint[resources[('primitive',member)],j] = 1
    # Reject double explanations of the same spatial proposal even if it came
    # from native content and template search through different proposal IDs.
    conflicts=[]
    for i,x in enumerate(pool):
        extent=np.array(x['bbox'][2:])-x['bbox'][:2]
        for j in spatial.query_ball_point(centers[i],.3*np.linalg.norm(extent)+2):
            if j<=i:
                continue
            y=pool[j]
            if set(x['classes']) & set(y['classes']) and box_iou(x['bbox'],y['bbox'])>.70:
                conflicts.append((x['id'],y['id']))
    if conflicts:
        overlap=lil_matrix((len(conflicts),len(hypotheses)),dtype=float)
        for k,(x,y) in enumerate(conflicts):
            for j,h in enumerate(hypotheses):
                if x in h['members'] or y in h['members']:
                    overlap[k,j]=1
        from scipy.sparse import vstack
        constraint=vstack([constraint,overlap],format='csc')
    solution=milp(np.array([h['cost']-.32 for h in hypotheses]),integrality=np.ones(len(hypotheses)),bounds=Bounds(0,1),constraints=LinearConstraint(constraint.tocsc(),0,1),options={'time_limit':30})
    if solution.x is None:
        raise RuntimeError('Automatic assignment has no feasible solver result')
    assignments={rid:{'state':'unresolved','reason':'no accepted automatic hypothesis'} for rid in slots}
    instances={};used=set()
    for j in np.flatnonzero(solution.x>.5):
        h=hypotheses[j];cid=f"auto{len(instances)}"
        alternatives=sorted(q['cost'] for q in hypotheses if q['reference_id']==h['reference_id'] and box_iou(q['bbox'],h['bbox'])<.5)
        margin=(alternatives[0]-h['cost']) if alternatives else None
        assignments[h['reference_id']]={'state':'matched','candidate_id':cid,'cost':h['cost'],
            'alternative_margin':margin,'ambiguous_alternative':margin is not None and margin<.03,
            'visibility_verified':False,'evidence_status':'localization_only','review_required':True}
        if h['class'] == 'text' and not text_verified.get(h['reference_id'], True):
            expected, observed = text_by_id.get(h['reference_id'], ''), h.get('text')
            assignments[h['reference_id']]['text_diagnostic'] = {
                'source_text_verified': False, 'source_text': expected, 'candidate_text': observed,
                'similarity': SequenceMatcher(None, ''.join(expected.split()), ''.join(observed.split())).ratio() if observed and expected is not None else None}
        instances[cid]={k:v for k,v in h.items() if k not in ('reference_id','cost')}
        instances[cid]['id']=cid;used.update(h['members'])
    return {'assignments':assignments,'instances':instances,'inventory_complete':False,
            'unclaimed_primitives':[q for q in primitives if q['id'] not in used],
            'diagnostics':{'hypotheses':len(hypotheses),'native_primitives':len(primitives),'raster_proposals':len(pool)-len(primitives),'solver_status':int(solution.status),'solver_gap':getattr(solution,'mip_gap',None)},
            'scope':'Automatic localization and grouping prototype. No candidate observations read. Supports, extra inventory, relation direction and rubric remain unverified; not a complete F ledger.'}


def resolve_observations(gt, reference, candidate, judgments):
    """One correspondence for L/G/A/T; unmatched visible instances determine H."""
    assert reference.size == candidate.size == tuple(gt["image_size"]), "Canvas size mismatch"
    expected = {
        "visual_contract_sha256": gt["visual_contract_sha256"],
        "reference_pixel_sha256": pixel_digest(reference),
        "candidate_pixel_sha256": pixel_digest(candidate),
        "correspondence_reference_sha256": hashlib.sha256(
            json.dumps(gt, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    for key, value in expected.items():
        if judgments[key] != value:
            raise ValueError("Observation/source mismatch: " + key)
    raw = judgments["correspondence"]
    for key in ("reference_pixel_sha256", "candidate_pixel_sha256"):
        if raw[key] != expected[key]:
            raise ValueError("Correspondence/image mismatch: " + key)
    digest = hashlib.sha256(
        json.dumps(
            raw, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    entities = gt["reference_entities"]
    slots = entities["instances"]
    instances, matches = dict(raw["instances"]), dict(raw["assignments"])
    owners = []
    for rid, match in matches.items():
        assert rid in slots and match["state"] in ("matched", "missing", "unresolved")
        cid = match.get("candidate_id")
        if cid is not None:
            assert cid in instances and instances[cid]["class"] == slots[rid]["class"]
            owners.append(cid)
        assert match["state"] != "matched" or cid is not None
        assert match["state"] != "missing" or cid is None
    assert len(owners) == len(set(owners)), (
        "A candidate instance cannot explain two source instances"
    )
    complete = raw["inventory_complete"] and all(
        matches.get(rid, {}).get("state") in ("matched", "missing") for rid in slots
    )
    counts = dict(
        gt["visual_contract"].get(
            "visible_class_counts", Counter(v["class"] for v in slots.values())
        )
    )
    for owner in (gt, gt["visual_contract"]):
        for kind, count in owner.get("extra_budget_counts", {}).items():
            assert counts.get(kind, 0) in (0, count)
            counts[kind] = count
    classes = sum(v > 0 for v in counts.values())
    extras = sorted(set(instances) - set(owners))
    budgets = [
        1 / classes / counts[instances[c]["class"]]
        if counts.get(instances[c]["class"], 0) > 0
        else None
        for c in extras
    ]
    H = sum(budgets) if complete and all(b is not None for b in budgets) else None

    # Aliases and compound anchors are references, not additional visible objects.
    for alias, target in entities["aliases"].items():
        matches[alias] = matches.get(target, {"state": "unresolved"})
    for rid, anchor in entities["anchors"].items():
        cid = "@source-anchor/" + rid
        if anchor["kind"] == "canvas":
            state, box = "matched", [0, 0, *reference.size]
        else:
            members = [matches.get(x, {"state": "unresolved"}) for x in anchor["members"]]
            present = [m for m in members if m["state"] == "matched"]
            state = (
                "unresolved"
                if any(m["state"] == "unresolved" for m in members)
                else ("matched" if present else "missing")
            )
            box = None
            if state == "matched":
                boxes = np.asarray([instances[m["candidate_id"]]["bbox"] for m in present])
                lo, hi = boxes[:, :2].min(0), boxes[:, 2:].max(0)
                source_boxes = np.asarray([slots[m]["bbox"] for m in anchor["members"]])
                source = np.r_[source_boxes[:, :2].min(0), source_boxes[:, 2:].max(0)]
                scale = (hi - lo) / (source[2:] - source[:2])
                box = np.r_[
                    lo + (np.asarray(anchor["bbox"][:2]) - source[:2]) * scale,
                    lo + (np.asarray(anchor["bbox"][2:]) - source[:2]) * scale,
                ].tolist()
        matches[rid] = {"state": state, "candidate_id": cid if state == "matched" else None}
        if state == "matched":
            instances[cid] = {"class": "anchor", "bbox": box}
    return {
        "matches": matches,
        "instances": instances,
        "reference_slots": slots,
        "H": H,
        "extras": extras,
        "ledger_digest": digest,
    }


def rubric_score(requirement, judgments, ledger):
    match, _ = assigned_instance(ledger, requirement["unit"])
    if match["state"] == "missing":
        return 0.0
    judgment = judgments["scores"].get(requirement["id"])
    if match["state"] != "matched" or judgment is None:
        return None
    assert judgment["candidate_id"] == match["candidate_id"], "Rubric instance mismatch"
    assert judgment["ledger_digest"] == ledger["ledger_digest"], "Rubric correspondence mismatch"
    assert type(judgment["level"]) is int and judgment["level"] in (0, 1, 2, 3)
    return judgment["level"] / 3


def missing_requirements(gt, components, relations, H):
    measured = {}
    for branch, leaves in components.items():
        for leaf in leaves:
            measured.setdefault((branch, leaf["unit"], leaf["family"]), []).append(leaf["score"])
    for row in relations:
        measured[("L_relations", row["unit"], "relation")] = [row["score"]]
    contract = gt["visual_contract"]
    required = list(contract["page_requirements"])
    for item in contract["inventory"]:
        required += [
            (branch, item["normalized_id"], family) for branch, family in item["requirements"]
        ]
    for text in gt["fullpage_text"]["text_regions"]:
        required += [
            (branch, text["id"], family)
            for branch, family in (("T", "content"), ("T", "typography"), ("L_layout", "layout"))
        ]
    missing = [
        key
        for key in sorted(set(map(tuple, required)))
        if key not in measured or any(v is None for v in measured[key])
    ]
    if H is None:
        missing.append(("H", "page", "extra_visible_instances"))
    return missing
