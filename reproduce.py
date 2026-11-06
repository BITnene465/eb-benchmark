"""Recompute all bundled examples and compare every score with the frozen release."""
import hashlib
import json
from pathlib import Path

from score import score

ROOT = Path(__file__).resolve().parent


def numeric_digest(result):
    # Packaging changes the implementation lock, not the scoring payload.
    return hashlib.sha256(json.dumps(
        {k: v for k, v in result.items() if k != "lock_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


if __name__ == "__main__":
    expected = json.loads((ROOT / "examples" / "expected.json").read_text())
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    for name, reference in expected.items():
        result = score(ROOT / "examples" / name / "case.json")
        assert numeric_digest(result) == reference["payload_sha256"], name
        (output / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        lo, hi = result["bounds"]["F"]
        print(f"{name:8s} F=[{lo:.6f}, {hi:.6f}] "
              f"measured={result['measured']}/{result['eligible']} PASS", flush=True)
    print("All four outputs match the frozen release, including every local score.")
