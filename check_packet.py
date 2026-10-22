"""Closed packet, fixed claim scopes, article consistency and privacy checks (no physics re-run).

Usage: python3 -B check_packet.py      (reproduce.py re-runs the calculations)
"""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
REPOSITORY = "https://github.com/tkmt-lab/hawking-dispersive-offset"
PAPER = "paper-hawking-b0-v1.2-en"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def tree():
    return [p for p in ROOT.rglob("*") if p.relative_to(ROOT).parts[0] != ".git"]


def fp(r):
    p = Path(r["path"])
    assert not p.is_absolute() and ".." not in p.parts, p
    q = ROOT / p
    assert q.is_file() and sha(q) == r["sha256"] and q.stat().st_size == r["bytes"], ("fingerprint", str(p))


def main():
    # 1. closed inventory
    rows = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in rows
        rows[name] = digest
        assert sha(ROOT / name) == digest, ("inventory hash", name)
    actual = {p.relative_to(ROOT).as_posix() for p in tree() if p.is_file()}
    assert actual == set(rows) | {"SHA256SUMS"}, sorted(actual ^ (set(rows) | {"SHA256SUMS"}))
    assert not any(p.is_symlink() for p in tree())
    assert not any("__pycache__" in p.parts or p.suffix in (".pyc", ".log", ".aux") for p in tree())
    for p in tree():
        if p.suffix == ".py":
            ast.parse(p.read_text(), filename=p.name)
    prov = load("PROVENANCE.json")
    assert prov["version"] == VERSION and prov["uploaded"] is False and prov["new_scattering_runs"] == 0
    assert {r["path"] for r in prov["files"]} == actual - {"SHA256SUMS", "PROVENANCE.json"}
    for r in prov["files"]:
        fp(r)
    for r in load("source-derivatives.json")["records"]:
        fp(r["public"])

    # 2. claims and scope
    claims = load("claims.json")
    assert claims["version"] == VERSION
    fp(claims["paper"])
    fp(claims["pdf"])
    assert claims["scope"] == dict(formal_asymptotic_calculation=True, assumptions_A1_A3_proved=False,
                                   remainder_bound=False, Lean_proof=False, human_expert_review=False,
                                   priority_established=False, independent_numerical_check_profiles=4,
                                   claims_error_in_prior_work=False), "unsupported scope promotion"
    types = set(claims["evidence_type_meaning"])
    for c in claims["claims"]:
        assert c["human_review"] is False and c["lean"] == []
        assert set(c["evidence_types"]) <= types and c["evidence"]
        for r in c["evidence"]:
            fp(r)
    by_id = {c["id"]: c for c in claims["claims"]}
    assert by_id["b0-formula"]["evidence_types"][0] == "formal-derivation-under-assumptions-A1-A3"
    assert "not a proof of A1-A3" in by_id["b0-independent-numerical"]["qualification"]

    # 3. article: labels, references, citations, wording, PDF record
    tex = (ROOT / f"{PAPER}.tex").read_text()
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    full = tex + "".join((ROOT / name).read_text() for name in inputs)
    labels = re.findall(r"\\label\{([^}]+)\}", full)
    assert len(labels) == len(set(labels)) and {c["paper_label"] for c in claims["claims"]} <= set(labels)
    assert set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", full)) <= set(labels)
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    cited = {k for g in re.findall(r"\\cite\{([^}]+)\}", tex) for k in g.split(",")}
    assert cited == bib, cited ^ bib
    for name in re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", tex):
        assert (ROOT / name).is_file(), name
    assert tex.count("VERIFICATION-RECORD.md") == 1 and (ROOT / "VERIFICATION-RECORD.md").is_file()
    assert r"\author{Tatsuhiko Takimoto\\\small Independent researcher}" in tex
    assert "Version 1.2" in tex and "unpublished" not in tex.lower()
    assert not re.search(r"\b(?:R\d{3}|T\d{4}|Claude|Codex|mailbox|coordinator|commit)\b", tex), "process label in article"
    assert not re.search(r"\b[0-9a-f]{40}\b", tex), "commit in article"
    assert "it does not prove A1--A3" in tex and "not a theorem" in tex
    pdf = load("pdf-validation.json")
    assert pdf["status"] == "PASS" and pdf["all_pages_viewed"] is True and pdf["pages"] == 15
    assert sha(ROOT / f"{PAPER}.pdf") == pdf["pdf_sha256"] and sha(ROOT / f"{PAPER}.tex") == pdf["source_sha256"]
    assert not pdf["overfull_boxes"] and not pdf["undefined_references"] and not pdf["unresolved_marks"]
    text = subprocess.run(["pdftotext", f"{PAPER}.pdf", "-"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout
    assert "??" not in text and "Version 1.2" in text

    # 4. citation metadata
    cff = load("CITATION.cff")
    author = [{"family-names": "Takimoto", "given-names": "Tatsuhiko", "affiliation": "Independent researcher"}]
    assert cff["version"] == VERSION and cff["repository-code"] == REPOSITORY and cff["license"] == "Apache-2.0"
    assert cff["authors"] == author and cff["preferred-citation"]["authors"] == author
    assert cff["preferred-citation"]["license"] == "CC-BY-4.0" and cff["preferred-citation"]["version"] == "1.2"
    zen = load(".zenodo.json")
    assert zen["version"] == VERSION and zen["license"] == "cc-by-4.0"
    assert zen["creators"] == [{"name": "Tatsuhiko Takimoto", "affiliation": "Independent researcher"}]
    assert any(r["identifier"] == REPOSITORY for r in zen["related_identifiers"])
    assert REPOSITORY in (ROOT / "README.md").read_text()

    # 5. stored numbers behind the headline statements
    ind = load("independent/analysis.json")["profiles"]
    sealed = load("independent/predictions-sealed.json")["profiles"]
    diffs = {k: ind[k]["b0_num"] - sealed[k]["b0"] for k in ("T05", "B4", "N1", "N2")}
    assert max(map(abs, diffs.values())) < 1.1e-7
    summ = load("b0/b0_summary.json")
    assert all(summ[k][c] for k in ("blind_erf", "blind_tcub") for c in ("crit1", "crit2", "crit3"))
    assert max(summ[k]["max_T0_diff_scattering_vs_zeromode"] for k in summ) < 1.3e-10

    # 6. privacy
    from publication_privacy import scan_tree
    privacy = scan_tree(ROOT)
    assert not privacy["hits"], privacy["hits"]
    print(json.dumps(dict(status="PASS", version=VERSION, files=len(actual), claims=len(claims["claims"]),
                          article_pages=pdf["pages"], privacy_hits=0,
                          scope="Static closure, claim scopes, article consistency and stored numbers; "
                                "reproduce.py re-runs the calculations.")))


if __name__ == "__main__":
    main()
