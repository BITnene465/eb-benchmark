"""LibreOffice render/save/reopen, using a temporary project-local profile."""

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pymupdf
from PIL import Image


def convert(source, directory, profile, format):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (source.stem + "." + format.split(":")[0])
    assert target != source, "Conversion must not overwrite the input PPTX"
    target.unlink(missing_ok=True)
    command = [
        shutil.which("soffice") or "soffice",
        "-env:UserInstallation=" + profile.as_uri(),
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--convert-to",
        format,
        "--outdir",
        str(directory),
        str(source),
    ]
    run = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    (directory / "libreoffice.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    assert target.is_file(), f"LibreOffice did not create {target}"
    return target


def render_pptx(source, directory, size, profile):
    options = {
        "UseLosslessCompression": {"type": "boolean", "value": "true"},
        "ReduceImageResolution": {"type": "boolean", "value": "false"},
    }
    pdf = convert(
        source,
        directory,
        profile,
        "pdf:impress_pdf_Export:" + json.dumps(options, separators=(",", ":")),
    )
    with pymupdf.open(pdf) as document:
        assert len(document) == 1, "Expected a single-slide PPTX"
        page = document[0]
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(size[0] / page.rect.width, size[1] / page.rect.height),
            alpha=False,
        )
        # MuPDF can round the raster extent up by one pixel.
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).crop((0, 0, *size))
        png = pdf.with_suffix(".png")
        image.save(png)
    return png


def validate_pptx(source, output, size):
    """V succeeds only after opening, rendering, saving and reopening one slide."""
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None, "Corrupt PPTX archive"
        slides = [
            n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")
        ]
        assert len(slides) == 1, "Expected a single-slide PPTX"
        ET.fromstring(archive.read(slides[0]))
    output = Path(output)
    runtime = Path(__file__).parent / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreoffice-", dir=runtime) as temporary:
        profile = Path(temporary) / "profile"
        original = render_pptx(source, output / "original", size, profile)
        saved = convert(source, output / "saved", profile, "pptx:Impress MS PowerPoint 2007 XML")
        reopened = render_pptx(saved, output / "reopened", size, profile)
    return {
        "valid": True,
        "rendered": str(original),
        "saved": str(saved),
        "reopened": str(reopened),
    }
