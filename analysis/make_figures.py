"""Generate the paper's three external figures from the archived run JSONs.

`body.tex` references `fig_pairs`, `fig_msmarco_500k` and `fig_dpr_490k` with
\\includegraphics, and none of them existed -- the paper could not compile. This
is the generator the artifact was missing.

Plain `analysis/make_figures.py --series` prints the series it would plot and
checks each against the paper, using numpy only; matplotlib is imported inside
the plotting function, so the data path is verifiable on a machine that has no
plotting library.

    python analysis/make_figures.py --runs-root RUNSDIR [...] --out paper/figs
    python analysis/make_figures.py --runs-root RUNSDIR [...] --series   # no plots

Every figure's series, and the sha256 of each input file, is written to
`figures.json` beside the PDFs, so a figure is traceable to the run that made it
in the same way a number is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# label -> (file pattern, the D whose curve to draw, colour, marker)
PAIRS = [
    ("e5-base-v2",        "pair_e5-base-v2.json",           3000, "#1f77b4", "o"),
    ("bge-small-en-v1.5", "pair_bge-small-en-v1.5.json",    3000, "#2ca02c", "s"),
    ("all-mpnet-base-v2", "pair_all-mpnet-base-v2.json",    3000, "#9467bd", "^"),
    ("all-MiniLM-L6-v2",  "pair_all-MiniLM-L6-v2.json",     3000, "#d62728", "D"),
    ("DPR ctx encoder",   "migration_dpr_490k.json",         898, "#7f7f7f", "v"),
]
KPRIMES = [10, 20, 50, 100, 200, 500, 1000, 2000]
BAR = 0.92


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find(roots, name: str) -> Path | None:
    """Locate an input by basename or by a run-dir-relative path tail."""
    want = name.replace("\\", "/")
    for root in roots:
        cand = root / Path(name)
        if cand.is_file():
            return cand
        hits = [p for p in root.rglob(Path(name).name)
                if str(p).replace("\\", "/").endswith(want)]
        if hits:
            return sorted(hits, key=lambda p: len(str(p)))[0]
    return None


def rows_at(obj, D):
    return [r for r in obj["frontier"] if abs(float(r.get("D", -1)) - D) < 0.5]


def series_pairs(roots):
    """recall@K' against K' for each pair, and the K' that first reaches the bar."""
    out, hashes, crossing = [], {}, {}
    for label, name, D, colour, marker in PAIRS:
        p = find(roots, name)
        if p is None:
            raise SystemExit(f"missing input for fig_pairs: {name}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = rows_at(obj, D)
        if not rows:
            raise SystemExit(f"{name}: no frontier rows at D={D}")
        rows.sort(key=lambda r: r["kprime"])
        xs = [r["kprime"] for r in rows]
        ys = [r["recall_at_kprime"] for r in rows]
        out.append(dict(label=label, x=xs, y=ys, D=D, marker=marker, colour=colour))
        hashes[p.name] = sha256(p)
        hit = next((x for x, y in zip(xs, ys) if y >= BAR), xs[-1])
        crossing[label] = hit
    return out, hashes, crossing


def series_frontier(roots, name, Ds):
    """savings% against recall@K', one curve per D."""
    p = find(roots, name)
    if p is None:
        raise SystemExit(f"missing input: {name}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    curves = []
    for D in Ds:
        rows = sorted(rows_at(obj, D), key=lambda r: r["kprime"])
        if not rows:
            continue
        curves.append(dict(
            D=D,
            x=[r["savings_pct"] for r in rows],
            y=[r["recall_at_kprime"] for r in rows],
            kprime=[r["kprime"] for r in rows]))
    # the equal-depth joins: recall at a fixed K' as savings fall with D
    joins = []
    for kp in KPRIMES:
        pts = []
        for c in curves:
            if kp in c["kprime"]:
                i = c["kprime"].index(kp)
                pts.append((c["x"][i], c["y"][i]))
        if len(pts) > 1:
            pts.sort()
            joins.append(dict(kprime=kp, x=[p[0] for p in pts], y=[p[1] for p in pts]))
    return dict(name=name, N=obj.get("N"), curves=curves, joins=joins,
                sha256=sha256(p))


def check(fig_pairs, ms, dpr, crossing):
    """The paper's stated values, re-checked against the files."""
    bad = []
    # `series_frontier` skips a D with no rows, so this list can be empty --
    # and it usually is: `migration`'s default --d-fracs is 1.0, so a 5,000
    # query run records D=5000 and never D=3000. An unguarded [0] then raises
    # IndexError and no figure is produced at all, which reads as a broken
    # tool rather than as missing input.
    want_d = 3000
    ms_at_d = [c for c in ms["curves"] if c["D"] == want_d]
    if not ms_at_d:
        have = sorted(c["D"] for c in ms["curves"])
        bad.append(f"MS MARCO has no curve at D={want_d}, so the paper's "
                   f"recall at K'=500 cannot be checked; D values present: "
                   f"{have}. Re-run `migration` with --d-fracs covering "
                   f"{want_d}, or point --check-d at one of these.")
    else:
        ms500 = ms_at_d[0]
        if 500 not in ms500["kprime"]:
            bad.append(f"MS MARCO curve at D={want_d} has no K'=500 point; "
                       f"depths present: {ms500['kprime']}")
        else:
            i = ms500["kprime"].index(500)
            if abs(ms500["y"][i] - 0.9295) > 5e-5:
                bad.append(f"MS MARCO recall at K'=500 is {ms500['y'][i]}, "
                           "paper says 0.9295")
    if ms["N"] != 500000:
        bad.append(f"MS MARCO N is {ms['N']}, paper says 500000")
    if dpr["N"] != 449071:
        bad.append(f"DPR N is {dpr['N']}, paper says 449071")
    want = {"e5-base-v2": 100, "bge-small-en-v1.5": 200,
            "all-mpnet-base-v2": 500, "all-MiniLM-L6-v2": 500, "DPR ctx encoder": 2000}
    for label, k in want.items():
        if crossing.get(label) != k:
            bad.append(f"{label}: K' for {BAR} is {crossing.get(label)}, paper says {k}")
    return bad


def plot(pairs, ms, dpr, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.grid(True, lw=0.3, alpha=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # fig_pairs
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for s in pairs:
        ax.plot(s["x"], s["y"], marker=s["marker"], ms=3, lw=1.1,
                color=s["colour"], label=s["label"])
    ax.axhline(BAR, ls="--", lw=0.8, color="k")
    ax.set_xscale("log")
    ax.set_xticks([10, 50, 200, 500, 2000])
    ax.set_xticklabels(["10", "50", "200", "500", "2000"])
    ax.set_xlabel("$K'$ (window depth)")
    ax.set_ylabel("cross-space recall")
    ax.set_ylim(0.55, 1.0)
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_pairs.pdf")
    plt.close(fig)

    # the two frontiers
    for data, fname in ((ms, "fig_msmarco_500k.pdf"), (dpr, "fig_dpr_490k.pdf")):
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        for c in data["curves"]:
            ax.plot(c["x"], c["y"], marker="o", ms=2.6, lw=1.0,
                    label=f"$D={c['D']:,}$")
        for j in data["joins"]:
            ax.plot(j["x"], j["y"], ls=":", lw=0.7, color="k")
        ax.set_xlabel("savings in forward passes (\\%)")
        ax.set_ylabel("cross-space recall")
        ax.legend(fontsize=5, frameon=False, loc="lower left")
        style(ax)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", required=True)
    ap.add_argument("--out", default=None, help="directory for the PDFs")
    ap.add_argument("--series", action="store_true",
                    help="print the series and check them; write no plots")
    a = ap.parse_args()
    roots = [Path(r) for r in a.runs_root]

    pairs, ph, crossing = series_pairs(roots)
    ms = series_frontier(roots, "20261013-165911_migration_msmarco_500k_bsl/migration.json",
                         [50, 200, 1000, 3000])
    dpr = series_frontier(roots, "migration_dpr_490k.json", [45, 180, 898])

    print("fig_pairs: K' that first reaches the %.2f bar" % BAR)
    for label, k in crossing.items():
        print(f"   {label:<20} K'={k}")
    print("\nfig_msmarco_500k: %d curves over D, %d equal-depth joins, N=%s"
          % (len(ms["curves"]), len(ms["joins"]), ms["N"]))
    for c in ms["curves"]:
        print("   D=%-6d " % c["D"] + "  ".join(
            f"K'={k}:{y:.4f}/{x:.1f}%" for k, x, y in
            zip(c["kprime"], c["x"], c["y"])))
    print("\nfig_dpr_490k: %d curves over D, N=%s" % (len(dpr["curves"]), dpr["N"]))
    for c in dpr["curves"]:
        print("   D=%-6d " % c["D"] + "  ".join(
            f"K'={k}:{y:.4f}/{x:.1f}%" for k, x, y in
            zip(c["kprime"], c["x"], c["y"])))

    bad = check(pairs, ms, dpr, crossing)
    for b in bad:
        print("   MISMATCH:", b)

    if not a.series and a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        try:
            plot(pairs, ms, dpr, out)
            print(f"\nwrote 3 PDFs to {out}")
        except ImportError as e:
            print(f"\nno plotting library ({e}); the series above are still "
                  f"checked. Install matplotlib and re-run without --series.")
            return 2
        payload = dict(pairs=pairs, pairs_crossing=crossing,
                       msmarco=ms, dpr=dpr, input_sha256=ph)
        # figures.json records what was plotted, so a figure traces to its run
        (out / "figures.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("\nseries checked;", "mismatches: %d" % len(bad) if bad else "all values agree")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
