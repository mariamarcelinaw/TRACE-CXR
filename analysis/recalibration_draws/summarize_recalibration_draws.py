"""Summarize draw-level local-recalibration results (replicate and primary run).
usage: summarize_draws.py REPLICATE_DRAW_CSV PRIMARY_DRAW_CSV OUT_EVIDENCE_DIR OUT_FIGURE_PNG"""
import sys, numpy as np, pandas as pd
rep_csv, prim_csv, ev_dir, fig_png = sys.argv[1:5]
Q = [('mean', np.mean), ('median', np.median), ('sd', lambda a: np.std(a, ddof=1)), ('min', np.min),
     ('p2_5', lambda a: np.percentile(a, 2.5)), ('p25', lambda a: np.percentile(a, 25)),
     ('p75', lambda a: np.percentile(a, 75)), ('p97_5', lambda a: np.percentile(a, 97.5)), ('max', np.max)]
def desc(prefix, a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return {f'{prefix}_{k}': (round(float(f(a)), 4) if len(a) > 1 else None) for k, f in Q}
r = pd.read_csv(rep_csv, low_memory=False)
r['ok'] = r.status.eq('SUCCESS')
rows = []
for (m, s, meth, n), g in r.groupby(['model', 'strategy', 'method', 'n'], sort=True):
    ok = g[g.ok]; e = dict(model=m, strategy=s, method=meth, n=int(n), attempted=len(g), successful=len(ok), failed=int((~g.ok).sum()))
    for c in ['ECE15', 'BA', 'Se', 'Sp']: e.update(desc(c, ok[c]))
    if meth in ('intercept_only', 'full_logistic'):
        e.update(desc('intercept', ok.intercept))
        if meth == 'full_logistic':
            e.update(desc('slope', ok.slope)); e['nonpositive_slopes'] = int((ok.slope <= 0).sum())
    rows.append(e)
pd.DataFrame(rows).to_csv(f'{ev_dir}/recalibration_draw_distribution.csv', index=False, lineterminator='\n')
# pre-registered poor-adaptation events P1-P6, paired with the frozen model on the same evaluation images
fz = r[r.method == 'frozen'][['model', 'strategy', 'n', 'draw_id', 'BA', 'ECE15', 'Brier', 'Se', 'Sp']]
mm = r[r.method != 'frozen'].merge(fz, on=['model', 'strategy', 'n', 'draw_id'], suffixes=('', '_frozen'), validate='many_to_one')
pr = []
for (m, s, meth, n), g in mm.groupby(['model', 'strategy', 'method', 'n'], sort=True):
    ok = g.ok; A = len(g)
    pr.append(dict(model=m, strategy=s, method=meth, n=int(n), attempted=A,
                   P1_BA_below_frozen=round(float(((g.BA < g.BA_frozen) & ok).sum() / A), 4),
                   P2_ECE15_above_frozen=round(float(((g.ECE15 > g.ECE15_frozen) & ok).sum() / A), 4),
                   P3_Brier_above_frozen=round(float(((g.Brier > g.Brier_frozen) & ok).sum() / A), 4),
                   P4_Se_below_frozen=round(float(((g.Se < g.Se_frozen) & ok).sum() / A), 4),
                   P5_Sp_below_frozen=round(float(((g.Sp < g.Sp_frozen) & ok).sum() / A), 4),
                   P6_fit_failure=round(float((~ok).sum() / A), 4),
                   ECE15_above_0_2=round(float(((g.ECE15 > 0.2) & ok).sum() / A), 4)))
pd.DataFrame(pr).to_csv(f'{ev_dir}/recalibration_poor_adaptation.csv', index=False, lineterminator='\n')
# primary run (source of Table IV): logistic recalibration only
p = pd.read_csv(prim_csv); prow = []
for (m, n), g in p.groupby(['model', 'n'], sort=True):
    e = dict(model=m, n=int(n), draws=len(g), convergence_warnings=0, max_lbfgs_iterations=int(g.n_iter.max()),
             nonpositive_slopes=int((g.slope <= 0).sum()))
    for c in ['ECE15', 'BA', 'Se', 'Sp', 'intercept', 'slope']: e.update(desc(c, g[c]))
    prow.append(e)
pd.DataFrame(prow).to_csv(f'{ev_dir}/recalibration_primary_run_distribution.csv', index=False, lineterminator='\n')
# figure: gate, primary run, spread over draws vs labeled budget, frozen reference
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
g = p[p.model == 'gated_fusion']; ns = sorted(g.n.unique()); x = np.arange(len(ns))
frozen = {'ECE15': 0.390, 'BA': 0.792, 'Se': 0.618, 'Sp': 0.966}
blue, ink, muted, grid = '#2a78d6', '#0b0b0b', '#52514e', '#e6e5e1'
fig, axs = plt.subplots(1, 4, figsize=(9, 2.9), facecolor='#fcfcfb')
for ax, c, t in zip(axs, ['ECE15', 'BA', 'Se', 'Sp'], ['ECE15 (lower is better)', 'Balanced accuracy', 'Sensitivity', 'Specificity']):
    ax.set_facecolor('#fcfcfb')
    for i, n in enumerate(ns):
        v = g[g.n == n][c].values; lo, q1, md, q3, hi = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        ax.plot([i, i], [lo, hi], color=blue, lw=2, solid_capstyle='round')
        ax.plot([i, i], [q1, q3], color=blue, lw=7, solid_capstyle='round', alpha=0.45)
        ax.plot(i, md, 'o', ms=6, color=blue, mec='#fcfcfb', mew=2)
    ax.axhline(frozen[c], color=muted, lw=1.2, ls=(0, (4, 3)))
    ax.text(len(ns) - 0.6, frozen[c], 'frozen', color=muted, fontsize=8, va='bottom', ha='right')
    ax.set_xticks(x, [str(n) for n in ns]); ax.set_xlim(-0.5, len(ns) - 0.5)
    ax.set_title(t, fontsize=9, color=ink, loc='left'); ax.set_xlabel('labeled Kermany images', color=muted, fontsize=8)
    ax.grid(axis='y', color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']: ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
fig.suptitle('Gated fusion, logistic recalibration over 500 stratified draws (primary run): median, 25-75th and 2.5-97.5th percentiles',
             fontsize=9, color=ink, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(fig_png, dpi=200, facecolor='#fcfcfb')
print('ok')
