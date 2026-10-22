"""Re-run the fast parts of the calculation and compare with the stored results.

Usage:  python3 -B reproduce.py [--output FILE] [--skip-exhibits]

Everything runs in a temporary copy of the packet; nothing in the packet is
modified.  Steps (about one minute in total on a laptop):

  1. exact local expansion (rational arithmetic, orders eps^3 and eps^6) and the
     g-form rewriting: outputs must equal the stored JSON exactly;
  2. the zero-frequency formula, Eq. (25), for the seven profiles of Table 1 at
     every stored cutoff: T(0), delta and the local b0, b1 against the stored values;
  3. two full scattering limits omega -> 0 (boundary-value solver) as a spot check;
  4. the curvature and low-frequency analyses re-run on the stored scans;
  5. all tables and figure data of the article rebuilt from stored results;
  6. the independent numerical check: its extrapolation re-run on its stored
     solves, the recorded predictions recomputed (exact rationals where possible)
     and compared with the independent values; then all 80 central solves of the
     independent code re-run with the installed libraries and extrapolated again.

Tolerances are stated next to each comparison.  They absorb last-bit differences
between numerical libraries; they are not error bounds of the physics.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, "-B"]
report = dict(status="RUNNING", steps={})


def load(p):
    return json.loads(Path(p).read_text())


def close(a, b, rel, abs_=0.0):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def deep_close(a, b, rel, path="$"):
    """Compare two JSON values; numbers within rel, everything else exactly.

    Fields named maxres_* are fit residuals at roundoff level (about 1e-12); their last
    digits depend on the platform's floating-point library, so they are compared with an
    absolute tolerance of 1e-10 instead (both values must stay below that level)."""
    if path.rsplit(".", 1)[-1].startswith("maxres_") and isinstance(a, float) and isinstance(b, float):
        assert abs(a) < 1e-10 and abs(b) < 1e-10, (path, a, b)
        return
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, sorted(set(a) ^ set(b))[:5])
        for k in a:
            deep_close(a[k], b[k], rel, f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            deep_close(x, y, rel, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            assert a is b, path
        else:
            assert close(float(a), float(b), rel, 1e-300) or (math.isnan(a) and math.isnan(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


def run(cmd, cwd, env=None):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    e.update(env or {})
    out = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    assert out.returncode == 0, (cmd, out.stderr[-3000:])
    return out.stdout


def step(name):
    def deco(fn):
        def wrapped(*a):
            t = time.time()
            res = fn(*a)
            report["steps"][name] = dict(status="PASS", seconds=round(time.time() - t, 2), **(res or {}))
            print(f"PASS {name} ({report['steps'][name]['seconds']} s)", flush=True)
        return wrapped
    return deco


@step("1-exact-local-expansion")
def exact(tmp):
    b0 = tmp / "b0"
    for order in ("3", "6"):
        (b0 / f"local_expansion_out_order{order}.json").unlink()
        run(PY + ["local_expansion.py"], b0, {"B0_ORDER": order})
        new = (b0 / f"local_expansion_out_order{order}.json").read_bytes()
        assert new == (ROOT / f"b0/local_expansion_out_order{order}.json").read_bytes(), order
    (b0 / "g_form_out.json").unlink()
    run(PY + ["g_form.py"], b0)
    assert (b0 / "g_form_out.json").read_bytes() == (ROOT / "b0/g_form_out.json").read_bytes()
    o6 = load(b0 / "local_expansion_out_order6.json")
    assert o6["b0_vmonomials"] == {"(3, 0, 0, 0, 0, 0)": "1/4", "(1, 1, 0, 0, 0, 0)": "-1/3",
                                   "(0, 0, 1, 0, 0, 0)": "1/12"}, o6["b0_vmonomials"]
    assert o6["nonzero_orders_T"] == [3, 6] and o6["matches_conjecture"] is True
    return dict(b0="(v'''' - 4 v'' v''' + 3 v''^3)/12", b1_terms=len(o6["b1_vmonomials"]),
                comparison="byte-identical JSON")


@step("2-zero-frequency-formula")
def zero_frequency(tmp):
    sys.path[:0] = [str(tmp / "b0"), str(tmp / "src")]
    import check_steps as cs
    from zero_mode import zero_mode
    stored = {**load(ROOT / "b0/check_known.json"), **load(ROOT / "b0/check_blind.json")}
    polys = {k: cs.load_poly(k) for k in ("b0_vmonomials", "b1_vmonomials")}
    n = 0
    worst = 0.0
    for key, rec in stored.items():
        prof = cs.PROFILES[key]()
        a = cs.taylor(prof)
        vd = {i: a[i] * math.factorial(i) for i in range(8)}
        assert close(cs.eval_poly(polys["b0_vmonomials"], vd), rec["b0_local"], 1e-9, 1e-12), key
        assert close(cs.eval_poly(polys["b1_vmonomials"], vd), rec["b1_local"], 1e-9, 1e-12), key
        for row in rec["per_Lam"]:
            z = zero_mode(prof, row["Lam"])
            for f in ("T0_over_TH", "delta", "G0_pred"):
                worst = max(worst, abs(z[f] - row["zero"][f]) / max(abs(row["zero"][f]), 1e-300) if f == "G0_pred"
                            else abs(z[f] - row["zero"][f]))
                assert close(z[f], row["zero"][f], 1e-9, 1e-13), (key, row["Lam"], f, z[f], row["zero"][f])
            n += 1
    return dict(profiles=len(stored), cases=n, tolerance="relative 1e-9 (absolute 1e-13)",
                max_abs_difference_T0_delta=worst)


@step("3-scattering-spot-check")
def scattering(tmp):
    import check_steps as cs
    stored = load(ROOT / "b0/check_blind.json")
    out = {}
    for key in ("blind_erf", "blind_tcub"):
        row = [r for r in stored[key]["per_Lam"] if r["Lam"] == 10.0][0]
        lim = cs.bvp_limit(cs.PROFILES[key](), 10.0)
        assert close(lim["T0_over_TH"], row["bvp"]["T0_over_TH"], 1e-9), (key, lim["T0_over_TH"])
        out[key] = dict(Lam=10, T0_over_TH=lim["T0_over_TH"], stored=row["bvp"]["T0_over_TH"])
    return dict(cases=out, tolerance="relative 1e-9")


@step("4-analyses-on-stored-scans")
def analyses(tmp):
    run(PY + ["analyze.py"], tmp / "src")
    for f in ("c_final.json", "c_summary.json"):
        deep_close(load(tmp / "results" / f), load(ROOT / "results" / f), 1e-8)
    run(PY + ["analyze_b0.py"], tmp / "b0")
    deep_close(load(tmp / "b0/b0_summary.json"), load(ROOT / "b0/b0_summary.json"), 1e-8)
    s = load(ROOT / "b0/b0_summary.json")
    assert all(s[k][c] for k in ("blind_erf", "blind_tcub") for c in ("crit1", "crit2", "crit3"))
    return dict(tolerance="relative 1e-8 on every number", prediction_first_criteria="all met")


@step("5-tables-and-figures")
def exhibits(tmp):
    out = tmp / "exhibits-rebuilt"
    run(PY + [str(tmp / "exhibits/build_exhibits.py"), "--out", str(out)], tmp)
    tex = sorted(p.name for p in (ROOT / "exhibits").glob("*.tex"))
    for name in tex:
        assert (out / name).read_bytes() == (ROOT / "exhibits" / name).read_bytes(), name
    assert (out / "all-profiles.md").read_bytes() == (ROOT / "exhibits/all-profiles.md").read_bytes()
    for name in ("figure-data-curvature.json", "figure-data-b0.json", "omega-window-sensitivity.json"):
        deep_close(load(out / name), load(ROOT / "exhibits" / name), 1e-8)
    figs = sorted(p.name for p in out.glob("*.pdf"))
    assert figs == sorted(p.name for p in (ROOT / "exhibits").glob("*.pdf"))
    return dict(tables_byte_identical=tex, figure_data="relative 1e-8", figures_rebuilt=figs)


def tanh_series(coeffs, n):
    """Taylor coefficients of tanh(p(x)) for a polynomial p with p(0) = 0, exact."""
    def mul(a, b):
        return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n)]
    c = {1: Fraction(1), 3: Fraction(-1, 3), 5: Fraction(2, 15), 7: Fraction(-17, 315)}
    p = (list(coeffs) + [Fraction(0)] * n)[:n]
    out, power = [Fraction(0)] * n, [Fraction(1)] + [Fraction(0)] * (n - 1)
    for k in range(1, n):
        power = mul(power, p)
        if k in c:
            out = [o + c[k] * q for o, q in zip(out, power)]
    return out


def b0_exact(amp, poly):
    t = tanh_series(poly, 6)
    vd = [amp * t[k] * math.factorial(k) for k in range(6)]
    assert vd[1] == 1
    return (vd[4] - 4 * vd[2] * vd[3] + 3 * vd[2] ** 3) / 12


@step("6-independent-check")
def independent(tmp):
    ind = tmp / "independent"
    run(PY + ["analyze.py"], ind)
    deep_close(load(ind / "analysis.json"), load(ROOT / "independent/analysis.json"), 1e-8)
    F = Fraction
    exact = {"T05": b0_exact(F(1, 2), [0, 2]),
             "B4": b0_exact(F(3, 5), [0, F(5, 3), F(-7, 10), F(1, 2)]),
             "N1": b0_exact(F(1, 2), [0, 2, F(2, 5), F(3, 10)])}
    u, w = 0.3 / 0.7, 0.7          # N2: derivatives of A[erf((x+s)/w) - erf(s/w)] with v'(0) = 1
    v2, v3, v4 = -2 * u / w, (4 * u * u - 2) / w ** 2, -(8 * u ** 3 - 12 * u) / w ** 3
    values = {k: float(v) for k, v in exact.items()}
    values["N2"] = (v4 - 4 * v2 * v3 + 3 * v2 ** 3) / 12
    sealed = load(ROOT / "independent/predictions-sealed.json")["profiles"]
    analysis = load(ROOT / "independent/analysis.json")["profiles"]
    rows = {}
    for k in ("T05", "B4", "N1", "N2"):
        assert close(values[k], sealed[k]["b0"], 1e-12, 1e-15), (k, values[k], sealed[k]["b0"])
        lo, hi = analysis[k]["combined_minmax"]
        rows[k] = dict(predicted=sealed[k]["b0"], exact=str(exact[k]) if k in exact else None,
                       independent=analysis[k]["b0_num"], difference=analysis[k]["b0_num"] - sealed[k]["b0"],
                       range_42=[lo, hi], prediction_in_range=lo <= sealed[k]["b0"] <= hi)
    assert max(abs(r["difference"]) for r in rows.values()) < 1.1e-7
    assert [k for k, r in rows.items() if not r["prediction_in_range"]] == ["T05"]
    assert 0 < -rows["T05"]["range_42"][1] < 6e-9
    # Replay all 80 central solves of the independent code with the installed libraries,
    # then its extrapolation on the replayed solves.
    replay = ind / "replay"
    replay.mkdir()
    for name in ("scatter.py", "analyze.py", "baseline.json", "dop853.json", "bvp-confirmation.json"):
        shutil.copyfile(ind / name, replay / name)
    run(PY + ["scatter.py", "--profiles", "T05", "B4", "N1", "N2", "--lambdas", "10", "20", "40", "80", "160",
              "--omegas", ".02", ".01", ".005", ".0025", "--step", ".5", "--extent", "9",
              "--output", "refined.json"], replay)
    new, old = load(replay / "refined.json"), load(ROOT / "independent/refined.json")
    assert [(r["profile"], r["Lambda"], r["omega"]) for r in new] == [(r["profile"], r["Lambda"], r["omega"]) for r in old]
    solve_diff = max(abs(a["T_over_TH_minus1"] - b["T_over_TH_minus1"]) for a, b in zip(new, old))
    assert solve_diff < 1e-10, solve_diff
    run(PY + ["analyze.py"], replay)
    rep = load(replay / "analysis.json")["profiles"]
    shift = {k: rep[k]["b0_num"] - analysis[k]["b0_num"] for k in rows}
    vs_pred = {k: rep[k]["b0_num"] - sealed[k]["b0"] for k in rows}
    assert max(map(abs, shift.values())) < 1e-7 and max(map(abs, vs_pred.values())) < 3e-7, (shift, vs_pred)
    return dict(profiles=rows, note="ranges are extrapolation sensitivity, not confidence intervals",
                replay=dict(max_abs_T_over_TH_minus1_change=solve_diff, b0_shift_from_stored=shift,
                            b0_minus_prediction=vs_pred,
                            note="library-dependent last bits are amplified by Lambda^2 and the extrapolation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    ap.add_argument("--skip-exhibits", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="hawking-b0-") as d:
        tmp = Path(d) / "packet"
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pdf"))
        for p in ROOT.glob("exhibits/*.pdf"):
            shutil.copyfile(p, tmp / "exhibits" / p.name)
        exact(tmp)
        zero_frequency(tmp)
        scattering(tmp)
        analyses(tmp)
        if not a.skip_exhibits:
            exhibits(tmp)
        independent(tmp)
    report.update(status="PASS", seconds=round(time.time() - t0, 1),
                  python=sys.version.split()[0])
    text = json.dumps(report, indent=1)
    if a.output:
        Path(a.output).write_text(text + "\n")
    print(json.dumps(dict(status="PASS", seconds=report["seconds"], steps=list(report["steps"]))))


if __name__ == "__main__":
    main()
