"""LaTeX table of mean RMSE and mean calibration error per compositional_bridge_d1 value, read from a
sweep_bridge_d1.py `sweep_summary.json`. Best value per column in bold. Writes `d1_sweep_table.tex` next to it.

    python scripts/d1_sweep_table.py <run>/sweep_d1/sweep_summary.json
"""
import json
import os
import sys

path = sys.argv[1]
m = json.load(open(path))["metrics"]
d1s = sorted(m, key=float)
rmse = [m[d]["rmse"]["mean"] for d in d1s]
cal = [m[d]["calibration_error"]["mean"] for d in d1s]


def bold(col):
    # best on the unrounded value, so a rounding tie (0.0296 vs 0.0304) does not bold both
    return [rf"\textbf{{{v:.3f}}}" if v == min(col) else f"{v:.3f}" for v in col]


rows = [rf"{float(d):g} & {r} & {c} \\" for d, r, c in zip(d1s, bold(rmse), bold(cal))]
tex = "\n".join([
    r"\begin{table}", r"\centering",
    r"\caption{Compositional posterior on the simulated test set as a function of the bridge parameter $d_1$: "
    r"RMSE (prior-std normalized) and calibration error, each averaged over the inferred global parameters. "
    r"Best value per column in bold.}",
    r"\label{tab:d1_sweep}",
    r"\begin{tabular}{lrr}", r"\hline",
    r"$d_1$ & RMSE & calibration \\", r"\hline",
    *rows, r"\hline", r"\end{tabular}", r"\end{table}", ""])
out = os.path.join(os.path.dirname(path), "d1_sweep_table.tex")
open(out, "w").write(tex)
print(tex)
