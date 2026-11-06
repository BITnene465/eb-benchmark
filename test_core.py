"""Small mathematical checks. Real-image regression uses the frozen 16 artifacts."""

import json
import unittest
import tempfile
import zipfile
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate import relation_score, evaluate
from data import automatic_correspondence, ppt_proposals, extra_evidence_budget
from geometry import transport, boundary_mass_local
from metrics import (
    content_score,
    glyph_evidence,
    layout_score,
    macro_score,
    visual_total,
    color_coverage,
)


class MetricChecks(unittest.TestCase):
    def test_unverified_source_ocr_does_not_become_text_ground_truth(self):
        gt = {'units': [], 'relations': [], 'visible_instance_supports': {},
              'rubric_requirements': [], 'complete': False,
              'visual_contract': {'page_requirements': [], 'inventory': []},
              'fullpage_text': {'text_regions': [{'id': 't', 'bbox': [1,1,9,9],
                                                'text': 'unreliable OCR', 'verified': False}]}}
        ledger = {'matches': {'t': {'state': 'matched', 'candidate_id': 'c'}},
                  'instances': {'c': {'bbox': [1,1,9,9], 'text': 'correct label'}},
                  'reference_slots': {'t': {'bbox': [1,1,9,9]}}, 'H': 0., 'extras': []}
        config = json.loads((Path(__file__).parent/'metric.json').read_text())
        image = Image.new('RGB', (10,10), 'white')
        with patch('evaluate.resolve_observations', return_value=ledger):
            for raw_ocr in ('unreliable OCR', 'correct label'):
                gt['fullpage_text']['text_regions'][0]['text'] = raw_ocr
                result = evaluate(gt, image, image, {}, config)
                content = next(x for x in result['components']['T'] if x['family']=='content')
                self.assertIsNone(content['score'])
                self.assertIn(('T','t','content'), result['missing'])
            gt['fullpage_text']['text_regions'][0]['verified'] = True
            result = evaluate(gt, image, image, {}, config)
            self.assertEqual(result['components']['T'][0]['score'], 1.)

    def test_color_area_and_unknown_budget(self):
        a=np.ones((10,10,3))*.8;b=a.copy();b[:3]=.4;w=np.ones((10,10))
        q=color_coverage(a[:3],b[:3],w[:3])['score']
        score=color_coverage(a,b,w)['score']
        self.assertAlmostEqual(score,.7+.3*q)
        known=np.ones((10,10),bool);known[:3]=False
        uncertain=color_coverage(a,b,w,known=known)
        self.assertIsNone(uncertain['score'])
        self.assertLessEqual(uncertain['lower'],score)
        self.assertGreaterEqual(uncertain['upper'],score)
        self.assertAlmostEqual(uncertain['upper']-uncertain['lower'],.3)

    def test_local_boundary_change_stays_in_filter_neighborhood(self):
        a=np.ones((40,80,3));a[10:30,10:65]=.7;b=a.copy();b[15:20,50:55]=0
        support=np.ones((40,80),bool)
        for mode in ('fixed','local'):
            first=boundary_mass_local(a,support,mode=mode)
            second=boundary_mass_local(b,support,mode=mode)
            np.testing.assert_array_equal(first[:,:40],second[:,:40])
            self.assertAlmostEqual(transport(first,first,[0])['score'],1)

    def test_extra_evidence_budget_is_split_and_duplicate_invariant(self):
        source = np.ones((4,6))
        extra = np.zeros_like(source);extra[1,1:4] = 1
        parts = []
        for x in (1,2,3):
            p = np.zeros_like(source);p[1,x] = 1;parts.append(p)
        a = extra_evidence_budget({'text':[source]}, {'text':[extra]})['H']
        b = extra_evidence_budget({'text':[source]}, {'text':parts})['H']
        c = extra_evidence_budget({'text':[source]}, {'text':[extra,extra]})['H']
        self.assertEqual(a,3/24)
        self.assertEqual(a,b)
        self.assertEqual(a,c)
        self.assertIsNone(extra_evidence_budget({'text':[source]}, {'graphic':[extra]})['H'])

    def test_explicit_invisible_shapes_are_not_native_proposals(self):
        ns_p = 'http://schemas.openxmlformats.org/presentationml/2006/main'
        ns_a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
        shapes = []
        for i, (fill, hidden) in enumerate([
            ('<a:noFill/>', ''),
            ('<a:solidFill><a:srgbClr val="000000"><a:alpha val="0"/></a:srgbClr></a:solidFill>', ''),
            ('<a:solidFill><a:srgbClr val="000000"/></a:solidFill>', ' hidden="1"'),
            ('<a:solidFill><a:srgbClr val="000000"/></a:solidFill>', ''),
        ]):
            shapes.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i+1}" name="x"{hidden}/></p:nvSpPr>'
                          '<p:spPr><a:xfrm><a:off x="10" y="10"/><a:ext cx="50" cy="30"/>'
                          f'</a:xfrm>{fill}<a:ln><a:noFill/></a:ln></p:spPr></p:sp>')
        runtime = Path(__file__).parent/'.runtime';runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            ppt = Path(directory)/'paint.pptx'
            with zipfile.ZipFile(ppt, 'w') as z:
                z.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{ns_p}"><p:sldSz cx="100" cy="80"/></p:presentation>')
                z.writestr('ppt/slides/slide1.xml', f'<p:sld xmlns:p="{ns_p}" xmlns:a="{ns_a}"><p:cSld><p:spTree>'+''.join(shapes)+'</p:spTree></p:cSld></p:sld>')
            proposals = ppt_proposals(ppt, (100,80))
        self.assertEqual(len(proposals),1)
        self.assertEqual(proposals[0]['bbox'],[10,10,60,40])

    def test_automatic_matching_groups_native_fragments(self):
        source=Image.new('RGB',(120,80),'white')
        ImageDraw.Draw(source).rectangle((20,20,100,60),fill='#427DA1')
        gt={'reference_entities':{'instances':{'s':{'class':'shape','bbox':[20,20,100,60]}}},'fullpage_text':{'text_regions':[]}}
        fragments=[{'id':'left','classes':['shape'],'bbox':[20,20,60,60],'origin':'native_geometry'},
                   {'id':'right','classes':['shape'],'bbox':[60,20,100,60],'origin':'native_geometry'}]
        with patch('data.ppt_proposals',return_value=fragments):
            out=automatic_correspondence(gt,source,source,'unused.pptx',ocr_regions=[])
        match=out['instances'][out['assignments']['s']['candidate_id']]
        self.assertEqual(set(match['members']),{'left','right'})
        self.assertEqual(match['bbox'],[20,20,100,60])
        self.assertTrue(out['assignments']['s']['review_required'])
        self.assertFalse(out['assignments']['s']['visibility_verified'])

    def test_automatic_matching_does_not_reuse_one_native_object(self):
        source=Image.new('RGB',(160,80),'white')
        draw=ImageDraw.Draw(source)
        draw.rectangle((10,20,50,60),fill='#427DA1');draw.rectangle((80,20,120,60),fill='#427DA1')
        candidate=Image.new('RGB',source.size,'white');ImageDraw.Draw(candidate).rectangle((45,20,85,60),fill='#427DA1')
        draw.line((12,22,48,58),fill='black',width=3);draw.line((82,22,118,58),fill='black',width=3)
        ImageDraw.Draw(candidate).line((47,22,83,58),fill='black',width=3)
        gt={'reference_entities':{'instances':{'a':{'class':'shape','bbox':[10,20,50,60]},'b':{'class':'shape','bbox':[80,20,120,60]}}},'fullpage_text':{'text_regions':[]}}
        one=[{'id':'only','classes':['shape'],'bbox':[45,20,85,60],'origin':'native_geometry'}]
        with patch('data.ppt_proposals',return_value=one):
            out=automatic_correspondence(gt,source,candidate,'unused.pptx',ocr_regions=[])
        self.assertEqual(sum(m['state']=='matched' for m in out['assignments'].values()),1)

    def test_automatic_matching_rejects_invisible_native_objects(self):
        source=Image.new('RGB',(120,80),'white');ImageDraw.Draw(source).rectangle((20,20,100,60),fill='#427DA1')
        gt={'reference_entities':{'instances':{'s':{'class':'shape','bbox':[20,20,100,60]}}},'fullpage_text':{'text_regions':[]}}
        one=[{'id':'hidden','classes':['shape'],'bbox':[20,20,100,60],'origin':'native_geometry'}]
        with patch('data.ppt_proposals',return_value=one):
            out=automatic_correspondence(gt,source,Image.new('RGB',source.size,'white'),'unused.pptx',ocr_regions=[])
        self.assertEqual(out['assignments']['s']['state'],'unresolved')

    def test_custom_arrow_requires_open_unfilled_path_and_arrow_end(self):
        ns_p = 'http://schemas.openxmlformats.org/presentationml/2006/main'
        ns_a = 'http://schemas.openxmlformats.org/drawingml/2006/main'
        shapes = []
        for i, (closed, filled, arrow) in enumerate([(False, False, True), (True, True, True), (False, True, True), (False, False, False)]):
            path = '<a:moveTo><a:pt x="0" y="0"/></a:moveTo><a:lnTo><a:pt x="50" y="30"/></a:lnTo>'
            path += '<a:close/>' if closed else ''
            fill = '<a:solidFill><a:srgbClr val="000000"/></a:solidFill>' if filled else '<a:noFill/>'
            end = '<a:tailEnd type="triangle"/>' if arrow else ''
            shapes.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{i+1}" name="object"/></p:nvSpPr><p:spPr>'
                f'<a:xfrm><a:off x="{i*60}" y="10"/><a:ext cx="50" cy="30"/></a:xfrm>'
                f'<a:custGeom><a:pathLst><a:path w="50" h="30" fill="{"norm" if filled else "none"}">{path}</a:path></a:pathLst></a:custGeom>'
                f'{fill}<a:ln><a:solidFill><a:srgbClr val="000000"/></a:solidFill>{end}</a:ln></p:spPr></p:sp>')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'.runtime') as directory:
            ppt = Path(directory)/'paths.pptx'
            with zipfile.ZipFile(ppt, 'w') as z:
                z.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{ns_p}"><p:sldSz cx="240" cy="80"/></p:presentation>')
                z.writestr('ppt/slides/slide1.xml', f'<p:sld xmlns:p="{ns_p}" xmlns:a="{ns_a}"><p:cSld><p:spTree>'+''.join(shapes)+'</p:spTree></p:cSld></p:sld>')
            proposals = ppt_proposals(ppt, (240,80))
        self.assertEqual([('connection' in q['classes']) for q in proposals], [True,False,False,False])

    def test_text_without_ocr_matches_visible_ink_but_not_blank(self):
        source=Image.new('RGB',(180,90),'white')
        ImageDraw.Draw(source).text((25,25),'ALPHA',fill='black',font_size=26)
        gt={'reference_entities':{'instances':{'t':{'class':'text','bbox':[23,25,120,55]}}},
            'fullpage_text':{'text_regions':[{'id':'t','text':'ALPHA','verified':True}]}}
        with patch('data.ppt_proposals',return_value=[]):
            same=automatic_correspondence(gt,source,source,'unused.pptx',ocr_regions=[])
            blank=automatic_correspondence(gt,source,Image.new('RGB',source.size,'white'),'unused.pptx',ocr_regions=[])
        self.assertEqual(same['assignments']['t']['state'],'matched')
        self.assertEqual(blank['assignments']['t']['state'],'unresolved')

    def test_raster_only_text_cannot_explain_two_source_words_with_one_ink(self):
        source=Image.new('RGB',(200,100),'white');candidate=Image.new('RGB',source.size,'white')
        for x in (10,100):ImageDraw.Draw(source).text((x,30),'ABC',fill='black',font_size=24)
        ImageDraw.Draw(candidate).text((55,30),'ABC',fill='black',font_size=24)
        gt={'reference_entities':{'instances':{'a':{'class':'text','bbox':[10,30,80,60]},'b':{'class':'text','bbox':[100,30,170,60]}}},
            'fullpage_text':{'text_regions':[{'id':k,'text':'ABC','verified':True} for k in ('a','b')]}}
        with patch('data.ppt_proposals',return_value=[]):
            result=automatic_correspondence(gt,source,candidate,'unused.pptx',ocr_regions=[])
        self.assertEqual(sum(a['state']=='matched' for a in result['assignments'].values()),1)

    def test_unverified_source_ocr_is_diagnostic_not_a_veto(self):
        source=Image.new('RGB',(200,110),'white');candidate=Image.new('RGB',source.size,'white')
        ImageDraw.Draw(source).text((20,20),'ALPHA',fill='black',font_size=26)
        ImageDraw.Draw(candidate).text((20,40),'ALPHA',fill='black',font_size=26)
        gt={'reference_entities':{'instances':{'t':{'class':'text','bbox':[18,20,115,50]}}},
            'fullpage_text':{'text_regions':[{'id':'t','text':'FRAAOSЖ#','verified':False}]}}
        native=[{'id':'word','classes':['text'],'bbox':[18,40,115,70],'origin':'native_text','text':'ALPHA'}]
        with patch('data.ppt_proposals',return_value=native):
            result=automatic_correspondence(gt,source,candidate,'unused.pptx',ocr_regions=[])
            blank=automatic_correspondence(gt,source,Image.new('RGB',source.size,'white'),'unused.pptx',ocr_regions=[])
        self.assertEqual(result['assignments']['t']['state'],'matched')
        self.assertFalse(result['assignments']['t']['text_diagnostic']['source_text_verified'])
        self.assertEqual(blank['assignments']['t']['state'],'unresolved')
        gt['fullpage_text']['text_regions'][0]['text'] = None
        with patch('data.ppt_proposals',return_value=native):
            unknown = automatic_correspondence(gt,source,candidate,'unused.pptx',ocr_regions=[])
        self.assertEqual(unknown['assignments']['t']['state'],'matched')
        self.assertIsNone(unknown['assignments']['t']['text_diagnostic']['source_text'])
        self.assertIsNone(unknown['assignments']['t']['text_diagnostic']['similarity'])

    def test_layout_scale_and_position(self):
        box = [0, 0, 30, 40]
        self.assertEqual(layout_score(box, box), 1)
        self.assertAlmostEqual(layout_score(box, [5, 0, 35, 40]), 0.75)
        self.assertEqual(layout_score(box, None), 0)

    def test_transport_keeps_duplicate_mass(self):
        a = np.zeros((3, 4))
        a[1, 1] = 1
        b = a.copy()
        self.assertEqual(transport(a, b, [0, 1])["score"], 1)
        b[1, 2] = 1
        self.assertAlmostEqual(transport(a, b, [0, 1])["score"], 2 / 3)
        self.assertEqual(transport(a, np.zeros_like(a), [1])["score"], 0)
        self.assertIsNone(transport(np.zeros_like(a), b, [1])["score"])

    def test_transport_is_symmetric_and_monotone_in_tolerance(self):
        a = np.zeros((4, 5))
        b = a.copy()
        a[1, 1], a[2, 2] = 0.75, 0.5
        b[1, 2], b[2, 3] = 0.5, 0.75
        forward = transport(a, b, [0, 1, 2])
        backward = transport(b, a, [0, 1, 2])
        self.assertEqual(forward["curve"], backward["curve"])
        self.assertEqual(forward["curve"], sorted(forward["curve"]))

    def test_macro_budgets_and_unknown(self):
        leaves = [
            {"family": "outline", "region": "page", "unit": "a", "score": 1},
            {"family": "corner", "region": "page", "unit": "a", "score": 0},
        ]
        self.assertEqual(macro_score(leaves)["score"], 0.5)
        # Additional samples of one feature family do not increase its budget.
        self.assertEqual(macro_score(leaves + [leaves[0]])["score"], 0.5)
        unknown = deepcopy(leaves)
        unknown[1]["score"] = None
        self.assertIsNone(macro_score(unknown)["score"])
        unknown[1]["state"] = "prediction_missing"
        self.assertEqual(macro_score(unknown)["score"], 0.5)

    def test_content_preserves_scientific_symbols(self):
        self.assertEqual(content_score("e\u0301  α", "é α"), 1)
        self.assertLess(content_score("x ≤ y", "x < y"), 1)

    def test_typography_duplicate_is_not_free(self):
        config = json.loads(Path(__file__).with_name("metric.json").read_text())["typography"]
        image = Image.new("RGB", (64, 32), "white")
        draw = ImageDraw.Draw(image)
        draw.text((2, 4), "ABC", fill="black", font_size=20)
        box = [0, 0, 64, 32]
        same = glyph_evidence(image, image, box, box, config)["score"]
        duplicate = image.copy()
        ImageDraw.Draw(duplicate).text((28, 4), "ABC", fill="black", font_size=20)
        extra = glyph_evidence(image, duplicate, box, box, config)["score"]
        self.assertAlmostEqual(same, 1, places=6)
        self.assertLess(extra, same)

    def test_relations_distinguish_bidirectional_from_forward(self):
        ledger = {
            "matches": {
                rid: {"state": "matched", "candidate_id": cid}
                for rid, cid in (("a", "ca"), ("b", "cb"), ("r", "cr"))
            },
            "instances": {
                "ca": {},
                "cb": {},
                "cr": {"endpoints": ["ca", "cb"], "direction": "forward", "continuous": True},
            },
        }
        relation = {"id": "r", "source": "a", "target": "b", "direction": "both"}
        self.assertEqual(relation_score(ledger, relation), 0)
        ledger["instances"]["cr"]["direction"] = "both"
        self.assertEqual(relation_score(ledger, relation), 1)
        ledger["matches"]["b"]["state"] = "unresolved"
        self.assertIsNone(relation_score(ledger, relation))

    def test_total_preserves_extra_budget_and_unknown(self):
        branches = dict.fromkeys(("L", "G", "A", "T"), 1.0)
        self.assertEqual(visual_total(branches), 100)
        self.assertEqual(visual_total(branches, 1), 50)
        branches["G"] = None
        self.assertIsNone(visual_total(branches))

    def test_relation_known_failure_survives_unobserved_other_conditions(self):
        ledger = {
            "matches": {rid: {"state": "matched", "candidate_id": "c"+rid} for rid in ("a","b","r")},
            "instances": {"ca": {}, "cb": {}, "cr": {}},
        }
        relation = {"id":"r", "source":"a", "target":"b", "direction":"forward"}
        # Full truth table: endpoint/direction validity AND continuity.
        for topology in (True, False, None):
            for continuity in (True, False, None):
                ledger["instances"]["cr"] = {
                    "endpoints": ["ca","cb"] if topology is not None else None,
                    "direction": "forward" if topology else "reverse",
                    "continuous": continuity,
                }
                expected = 0.0 if False in (topology,continuity) else (None if None in (topology,continuity) else 1.0)
                self.assertEqual(relation_score(ledger,relation),expected)
        ledger["matches"]["b"]["state"]="unresolved"
        ledger["instances"]["cr"]["continuous"]=False
        self.assertEqual(relation_score(ledger,relation),0.0)
        ledger["instances"]["cr"]["continuous"]=None
        self.assertIsNone(relation_score(ledger,relation))
        ledger["matches"]["b"]["state"]="missing"
        self.assertEqual(relation_score(ledger,relation),0.0)


if __name__ == "__main__":
    unittest.main()
