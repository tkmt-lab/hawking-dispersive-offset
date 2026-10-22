"""Rebuild every table and figure of the article from the stored result files.

Usage: python3 -B exhibits/build_exhibits.py [--out DIR]   (default: a fresh temporary directory)

Reads results/*.json and b0/*.json only.  It does not import a solver, run a
scattering or zero-frequency calculation, or refit anything except the
documented three-point frequency-window refits of stored solves (part 3).
Writes the TeX tables, the three vector figures and the figure data.  The
check in reproduce.py compares the TeX tables byte for byte and the figure
data numerically with the copies in exhibits/.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
B0 = ROOT / "b0"
parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None)
args = parser.parse_args()
OUT = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="exhibits-"))
OUT.mkdir(parents=True, exist_ok=True)


def read(name):
    return json.loads((RESULTS / name).read_text())


# ---- part 1: curvature coefficient c (Sections 3-6) ----
final = read("c_final.json")
summary = read("c_summary.json")
predictions = read("c_predictions_preregistered.json")
predictions.update(read("c_predictions_blind_stage2.json"))
keys = ["tanh05", "tanh10"] + [f"flat_m{m}" for m in range(1, 5)] + [
    "sh_q2_tm025", "sh_q2_tm0125", "sh_q2_tp0125", "sh_q2_tp025",
    "sh_q4_tm0125", "sh_q4_tp0125", "sum_asym1", "sum_asym2",
    "sum_mix3", "sum_mix4", "sum_mix5", "blind_a", "blind_b",
]
labels = {
    "tanh05": r"T_{0.5}", "tanh10": r"T_1",
    **{f"flat_m{m}": f"F_{m}" for m in range(1, 5)},
    "sh_q2_tm025": r"S_{2,-1/4}", "sh_q2_tm0125": r"S_{2,-1/8}",
    "sh_q2_tp0125": r"S_{2,1/8}", "sh_q2_tp025": r"S_{2,1/4}",
    "sh_q4_tm0125": r"S_{4,-1/8}", "sh_q4_tp0125": r"S_{4,1/8}",
    "sum_asym1": r"A_+", "sum_asym2": r"A_-",
    "sum_mix3": r"M_3", "sum_mix4": r"M_4", "sum_mix5": r"M_5",
    "blind_a": r"B_1", "blind_b": r"B_2",
}

bindings = []


def cell(filename, pointer, value, fmt="+.6f"):
    text = format(value, fmt)
    if float(text) == 0:
        text = format(0.0, fmt)
    bindings.append(dict(file="results/" + filename, pointer=pointer,
                         format=fmt, rendered=text))
    return text


def final_cell(key, field, fmt="+.6f"):
    return cell("c_final.json", [key, field], final[key][field], fmt)


table = []
selected = ["tanh05", "tanh10", "flat_m4", "sh_q2_tp0125",
            "sh_q4_tp0125", "sum_asym1", "blind_a", "blind_b"]
for key in selected:
    row = ["$" + labels[key] + "$", final_cell(key, "Lam_max", ".0f"),
           final_cell(key, "c_inf"), final_cell(key, "c_formula"),
           final_cell(key, "c_DPLS")]
    table.append(" & ".join([row[0]] + ["$" + v + "$" for v in row[1:]]) + r" \\")
(OUT / "curvature-table.tex").write_text(
    "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
    r"Profile & $\Lambda_{\max}$ & $\cext$ & $\cphase$ & $\cefh$ \\" +
    "\n\\midrule\n" + "\n".join(table) + "\n\\bottomrule\n\\end{tabular}\n")

table = []
for key in ["blind_a", "blind_b"]:
    row = ["$" + labels[key] + "$", final_cell(key, "b0_at_Lam_max"),
           final_cell(key, "b0_inf"), final_cell(key, "b0_guess")]
    table.append(" & ".join([row[0]] + ["$" + v + "$" for v in row[1:]]) + r" \\")
(OUT / "intercept-table.tex").write_text(
    "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
    r"Profile & $b_0(320)$ & $\bext$ & $\bcand$ \\" +
    "\n\\midrule\n" + "\n".join(table) + "\n\\bottomrule\n\\end{tabular}\n")

table = []
for key in ["sum_asym1", "sum_asym2", "sum_mix3", "sum_mix4", "sum_mix5", "blind_a", "blind_b"]:
    filename = "c_predictions_blind_stage2.json" if key.startswith("blind") else "c_predictions_preregistered.json"
    comps = []
    for i, triple in enumerate(predictions[key]["comps"]):
        fields = [cell(filename, [key, "comps", i, j], value, ".3g")
                  for j, value in enumerate(triple)]
        comps.append("(" + ",".join(fields) + ")")
    table.append("$" + labels[key] + "$ & $" + r";\quad ".join(comps) + r"$ \\")
(OUT / "profile-table.tex").write_text(
    "\\begin{tabular}{@{}ll@{}}\n\\toprule\n"
    r"Profile & Triples $(h_i,q_i,t_i)$ \\" +
    "\n\\midrule\n" + "\n".join(table) + "\n\\bottomrule\n\\end{tabular}\n")

# Make all numerical tables available in a compact record, with no extra fits.
lines = ["| Profile | Input key | c extrapolated | c phase | c effective horizon | b0 extrapolated |",
         "|---|---|---:|---:|---:|---:|"]
for k in keys:
    f = final[k]
    lines.append(f"| ${labels[k]}$ | `{k}` | {f['c_inf']:+.9f} | {f['c_formula']:+.9f} | {f['c_DPLS']:+.9f} | {f['b0_inf']:+.9f} |")
(OUT / "all-profiles.md").write_text("\n".join(lines) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, (ax, residual) = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True,
                                  gridspec_kw={"width_ratios": [2.6, 1.25]}, layout="constrained")
y = np.arange(len(keys))
measured = np.array([final[k]["c_inf"] for k in keys])
phase = np.array([final[k]["c_formula"] for k in keys])
efh = np.array([final[k]["c_DPLS"] for k in keys])
phase_artist = ax.scatter(phase, y, marker="o", facecolors="none", edgecolors="#147d78", s=47, label=r"$c_{\rm phase}$", zorder=2)
measured_artist = ax.scatter(measured, y, marker="x", c="#152f4f", s=24, label=r"$c_{\rm ext}$", zorder=3)
efh_artist = ax.scatter(efh, y, marker="|", c="#b45b27", s=70, label=r"$c_{\rm EFH}$", zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(["$" + labels[k] + "$" for k in keys])
ax.invert_yaxis()
ax.set_xlabel("Frequency-curvature coefficient")
ax.axvline(0, color="0.8", lw=0.7, zorder=0)
ax.grid(axis="y", color="0.93", lw=0.5)
ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=1)
ax.set_xlim(-18.3, 20.5)
residual_artist = residual.scatter((measured-phase)*1000, y, c="#152f4f", marker=".", s=27)
residual.axvline(0, color="#147d78", lw=0.9)
residual.grid(axis="y", color="0.93", lw=0.5)
residual.set_xlabel(r"$10^3(c_{\rm ext}-c_{\rm phase})$")
residual.set_xlim(-1.6, 1.6)
fig.savefig(OUT / "curvature-comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
figure_data = {"comparison": dict(keys=keys, phase=phase_artist.get_offsets().tolist(),
                                  measured=measured_artist.get_offsets().tolist(),
                                  effective_horizon=efh_artist.get_offsets().tolist(),
                                  residual=residual_artist.get_offsets().tolist()),
               "convergence": []}
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.9), layout="constrained")
for ax, k in zip(axes.flat, ["sum_mix3", "sum_mix5", "blind_a", "blind_b"]):
    per = summary[k]["by_def"]["T_row"]["per_Lam"]
    lams = np.array([p["Lam"] for p in per])
    line, = ax.plot(lams, [p["c_A"] for p in per], "o-", color="#152f4f", ms=3.8, lw=1.2)
    ax.axhline(final[k]["c_formula"], color="#147d78", ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 80, 160, 320])
    ax.set_xticklabels([10, 20, 40, 80, 160, 320], fontsize=8)
    ax.set_title("$" + labels[k] + "$", fontsize=10, loc="left", pad=2)
    ax.set_xlabel(r"$\Lambda/\kappa$", labelpad=2)
    ax.set_ylabel(r"$c(\Lambda)$")
    ax.grid(alpha=.15)
    figure_data["convergence"].append(dict(key=k, points=line.get_xydata().tolist(),
                                           phase_line=final[k]["c_formula"]))
fig.savefig(OUT / "curvature-convergence.pdf", metadata={"CreationDate": None, "ModDate": None})
plt.close(fig)
(OUT / "figure-data-curvature.json").write_text(json.dumps(figure_data, indent=2) + "\n")

# ---- part 2: low-frequency offset b0 (Section 7) ----
def read_b0(name):
    return json.loads((B0 / name).read_text())


summary = read_b0("b0_summary.json")
known = read_b0("check_known.json")
blind = read_b0("check_blind.json")
checks = {**known, **blind}
b1fit = read_b0("b1_check.json")
pred = read_b0("blind_predictions.json")

keys = ["tanh05", "sh_q2_tp025", "sum_asym1", "sum_mix4", "blind_b",
        "blind_erf", "blind_tcub"]
labels = {"tanh05": r"T_{0.5}", "sh_q2_tp025": r"S_{2,1/4}",
          "sum_asym1": r"A_+", "sum_mix4": r"M_4", "blind_b": r"B_2",
          "blind_erf": r"B_3", "blind_tcub": r"B_4"}

bindings = []


def cell(filename, pointer, value, fmt):
    text = format(value, fmt)
    if float(text) == 0:
        text = format(0.0, fmt)
    bindings.append(dict(file="b0/" + filename, pointer=pointer, format=fmt,
                         rendered=text, value=value))
    return text


def sci(x):
    """LaTeX for a one-significant-figure magnitude, e.g. 1.0\\times10^{-7}."""
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


rows = []
derived = {}
for key in keys:
    S = summary[key]
    per = [r for r in checks[key]["per_Lam"] if "bvp" in r]
    maxdiff = max(abs(r["bvp"]["T0_over_TH"] - r["zero"]["T0_over_TH"]) for r in per)
    assert abs(maxdiff - S["max_T0_diff_scattering_vs_zeromode"]) < 1e-20
    b1 = b1fit[key]["fits"]["Lam>=28"]
    diff = S["b0_ext_scattering"] - S["b0_pred"]
    derived[key] = dict(diff=diff, maxdiff=maxdiff, b1_rel=abs(b1 / S["b1_pred"] - 1),
                        max_unit_err=max(r["bvp"]["max_unit_err"] for r in per),
                        max_rel_G0=S["max_rel_G0_diff"])
    row = ["$" + labels[key] + "$",
           "$" + cell("b0_summary.json", [key, "b0_pred"], S["b0_pred"], "+.6f") + "$",
           "$" + cell("b0_summary.json", [key, "b0_ext_scattering"], S["b0_ext_scattering"], "+.6f") + "$",
           "$" + ("+" if diff >= 0 else "-") + sci(abs(diff)) + "$",
           "$" + sci(maxdiff) + "$",
           "$" + cell("b0_summary.json", [key, "b1_pred"], S["b1_pred"], "+.5f") + "$",
           "$" + cell("b1_check.json", [key, "fits", "Lam>=28"], b1, "+.5f") + "$"]
    rows.append(" & ".join(row) + r" \\")
    if key == "blind_b":
        rows.append(r"\midrule")
(OUT / "b0-table.tex").write_text(
    "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n"
    r"Profile & $b_0$ (Eq.~\eqref{eq:b0}) & $b_0$ from scattering & difference"
    r" & $\max|\Delta T(0)|/\HawkingT$ & $b_1$ (local) & $b_1$ (fit) \\" +
    "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# Figure: blind profiles, Lambda^2 (T(0)/T_H - 1) against Lambda^-2.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
NAVY, TEAL, GREY = "#152f4f", "#147d78", "0.55"
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.75), layout="constrained",
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})
figdata = {"blind_panels": [], "difference_panel": []}
for ax, key in zip(axes[:2], ["blind_erf", "blind_tcub"]):
    b0p, b1p = checks[key]["b0_local"], checks[key]["b1_local"]
    per = checks[key]["per_Lam"]
    zs = [(r["Lam"] ** -2, r["Lam"] ** 2 * (r["zero"]["T0_over_TH"] - 1)) for r in per]
    bs = [(r["Lam"] ** -2, r["Lam"] ** 2 * (r["bvp"]["T0_over_TH"] - 1)) for r in per if "bvp" in r]
    zs = [(x, abs(y - b0p)) for x, y in zs]
    bs = [(x, abs(y - b0p)) for x, y in bs]
    xs = [min(x for x, _ in zs) / 1.3, max(x for x, _ in zs) * 1.3]
    ax.plot(xs, [abs(b1p) * x for x in xs], "--", color=TEAL, lw=1.2,
            label=r"$|b_1|\,\Lambda^{-2}$ (predicted)")
    ax.scatter(*zip(*zs), marker="o", facecolors="none", edgecolors=GREY, s=38,
               label="zero-frequency formula", zorder=2)
    ax.scatter(*zip(*bs), marker="x", c=NAVY, s=26, label=r"scattering, $\omega\to0$", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("$" + labels[key] + "$", fontsize=9)
    ax.set_xlabel(r"$\Lambda^{-2}$")
    ax.grid(color="0.93", lw=0.5)
    figdata["blind_panels"].append(dict(key=key, b0_pred=b0p, b1_pred=b1p, zero=zs, scattering=bs,
                                        extrapolation=summary[key]["b0_ext_scattering"]))
axes[0].set_ylabel(r"$|\Lambda^2[T(0)/T_{\rm H}-1]-b_0|$")
handles, labs = axes[0].get_legend_handles_labels()
ax = axes[2]
for i, key in enumerate(keys):
    d = abs(derived[key]["diff"])
    isblind = summary[key]["blind"]
    ax.scatter([d], [i], marker="o", s=30,
               facecolors=NAVY if isblind else "none", edgecolors=NAVY, zorder=3)
    figdata["difference_panel"].append([key, d, i, isblind])
ax.set_xscale("log")
ax.set_yticks(range(len(keys)), ["$" + labels[k] + "$" for k in keys])
ax.invert_yaxis()
ax.set_xlabel(r"$|b_0^{\rm scatt}-b_0|$")
ax.set_title("all seven profiles", fontsize=9)
ax.grid(axis="x", color="0.93", lw=0.5)
fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.1))
fig.savefig(OUT / "b0-comparison.pdf", metadata={"CreationDate": None, "ModDate": None},
            bbox_inches="tight")
(OUT / "figure-data-b0.json").write_text(json.dumps(figdata, indent=1) + "\n")


# ---- part 3: frequency-window sensitivity and the b1 polynomial (Sections 7.4, A) ----
checks = {**json.loads((B0 / "check_known.json").read_text()),
          **json.loads((B0 / "check_blind.json").read_text())}
keys = ["tanh05", "sh_q2_tp025", "sum_asym1", "sum_mix4", "blind_b", "blind_erf", "blind_tcub"]

# 1. sensitivity
def fit_T0(raw, idx):
    W = np.array([raw[i][0] for i in idx]); T = np.array([raw[i][1] for i in idx])
    A = np.vstack([W ** (2 * j) for j in range(len(idx))]).T
    return float(np.linalg.solve(A, T)[0])

sens = {"cases": []}
for k in keys:
    for r in checks[k]["per_Lam"]:
        if "bvp" not in r:
            continue
        raw, z = r["bvp"]["raw"], r["zero"]["T0_over_TH"]
        all4 = fit_T0(raw, [0, 1, 2, 3])
        assert abs(all4 - r["bvp"]["T0_over_TH"]) < 1e-13
        small3 = fit_T0(raw, [1, 2, 3])
        large3 = fit_T0(raw, [0, 1, 2])
        sens["cases"].append(dict(key=k, Lam=r["Lam"], diff_all4=all4 - z, diff_small3=small3 - z,
                                  diff_large3=large3 - z, smallest_omega_minus_zero=raw[3][1] - z))
c = sens["cases"]
sens["max_abs"] = {f: max(abs(x[f]) for x in c) for f in
                   ("diff_all4", "diff_small3", "diff_large3", "smallest_omega_minus_zero")}
sens["max_abs_by_window_change"] = max(max(abs(x["diff_small3"] - x["diff_all4"]),
                                           abs(x["diff_large3"] - x["diff_all4"])) for x in c)
# b0 extrapolation of the blind cases with the alternative windows
def b0_ext(k, field):
    per = {x["Lam"]: x for x in c if x["key"] == k}
    zero = {r["Lam"]: r["zero"]["T0_over_TH"] for r in checks[k]["per_Lam"]}
    Ls = np.array([20.0, 40.0, 80.0])
    y = np.array([L * L * (zero[L] + per[L][field] - 1) for L in Ls])
    return float(np.linalg.solve(np.vstack([np.ones(3), Ls ** -2, Ls ** -4]).T, y)[0])
sens["b0_ext_by_window"] = {k: {f: b0_ext(k, f) - checks[k]["b0_local"]
                                for f in ("diff_all4", "diff_small3", "diff_large3")}
                            for k in ("blind_erf", "blind_tcub")}
(OUT / "omega-window-sensitivity.json").write_text(json.dumps(sens, indent=1) + "\n")

# 2. b1 polynomial
gf = json.loads((B0 / "g_form_out.json").read_text())
terms = [t.strip() for t in gf["b1_gform"].split(" + ")]
parsed = []
for t in terms:
    coef, *mons = t.split("*")
    parsed.append((coef, mons))
assert len(parsed) == 18


def b1_from_g(gv):
    tot = 0.0
    for coef, mons in parsed:
        num, _, den = coef.partition("/")
        val = float(num) / (float(den) if den else 1.0)
        for m in mons:
            name, _, p = m.partition("^")
            val *= gv[int(name[1:])] ** (int(p) if p else 1)
        tot += val
    return tot


def vders_from_g(gv):
    """v^(n)(0), n = 1..7, for u' = g(u) = sum gv[j] u^j (gv[0] = 1)."""
    poly = np.array(gv[:7], dtype=float)  # coefficients of g in u
    cur = poly.copy(); out = [cur[0]]
    for _ in range(6):
        d = np.array([j * cur[j] for j in range(1, len(cur))] + [0.0])
        cur = np.convolve(poly, d)[:7]
        out.append(cur[0])
    return out


pred = json.loads((B0 / "blind_predictions.json").read_text())
b1check = {}
for k in ("blind_erf", "blind_tcub"):
    vd = pred[k]["vderivs"]  # v, v', ..., v^(7)
    sol = fsolve(lambda g: np.array(vders_from_g([1.0, *g])[1:]) - np.array(vd[2:8]),
                 np.zeros(6), xtol=1e-14)
    gv = [1.0, *sol]
    b1check[k] = dict(b1_gform=b1_from_g(gv), b1_pred=pred[k]["b1"], g3_over2=gv[3] / 2, b0_pred=pred[k]["b0"])
    assert abs(b1check[k]["b1_gform"] - pred[k]["b1"]) < 1e-8 * max(1, abs(pred[k]["b1"])), b1check[k]
    assert abs(b1check[k]["g3_over2"] - pred[k]["b0"]) < 1e-9
# T_{0.5}: g = 1 - 4u^2, b1 = 1/4
assert abs(b1_from_g([1, 0, -4, 0, 0, 0, 0]) - 0.25) < 1e-15


def tex_term(coef, mons):
    num, _, den = coef.partition("/")
    sign = "-" if num.startswith("-") else "+"
    num = num.lstrip("-")
    c = (rf"\tfrac{{{num}}}{{{den}}}" if den else ("" if num == "1" and mons else num))
    m = "".join(re.sub(r"g(\d)(?:\^(\d))?", lambda x: f"g_{x.group(1)}" + (f"^{x.group(2)}" if x.group(2) else ""), mm)
                for mm in mons)
    return sign, c + m


def order(t):  # sort: by total weight then lexicographic
    coef, mons = t
    w = sum(int(m[1]) * (int(m.partition("^")[2]) if "^" in m else 1) for m in mons)
    return (-w, mons)


pieces = [tex_term(*t) for t in sorted(parsed, key=order)]
lines, cur = [], ""
for i, (s, body) in enumerate(pieces):
    s_ = ("" if s == "+" else "-") if i == 0 else s
    cur += f"{s_}{body}"
    if (i + 1) % 6 == 0 and i + 1 < len(pieces):
        lines.append(cur); cur = ""
lines.append(cur)
(OUT / "b1-polynomial.tex").write_text(
    "\\begin{align}\n b_1={}&" + "\\nonumber\\\\\n &".join(lines) + ".\n \\label{eq:b1}\n\\end{align}\n")
(OUT / "extras-audit.json").write_text(json.dumps(dict(b1_check=b1check, n_terms=len(parsed)), indent=1) + "\n")
print(json.dumps(dict(out=str(OUT), files=sorted(p.name for p in OUT.iterdir())), indent=1))
