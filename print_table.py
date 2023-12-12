from pylatex import Document, Section, Subsection, Tabular, MultiColumn,\
    MultiRow, NoEscape
from pylatex.math import Math
from collections import OrderedDict
import copy
import numpy as np

from modelguidedattacks.results import build_full_results_dict

def result_to_str(result, long=False):
    if result is None or np.isinf(result) or np.isnan(result):
        return "-"
    elif long:
        return f"{result:.4f}"
    else:
        return f"{result:.2f}"

"""
results top level will be keyed by K
next level will be binary search steps
next level will be keyed by iterations
next level will be keyed by method
"""

only_mean = False
model_name = "resnet50"
results = build_full_results_dict(model_name)

model_to_tex = {
    "resnet50": "Resnet-50",
    "densenet121": "Densenet121",
    "deit_small": "DeiT-S",
    "vit_base": "ViT$_{B}$"
}

# Preprocess all results and select bests
for top_k, bs_dict in results.items():
    for num_bs, iter_dict in bs_dict.items():
        for num_iter, method_dict in iter_dict.items():
            metric_bests = {}
            metrics_compared = {}

            for method_name, method_results in method_dict.items():
                for metric_name, metric_value in method_results.items():
                    reduction_func = max if "ASR" in metric_name else min

                    if metric_name not in metric_bests:
                        metric_bests[metric_name] = 0. if reduction_func is max else np.Infinity
                        metrics_compared[metric_name] = 0

                    if metric_value is not None:
                        metric_bests[metric_name] = reduction_func(metric_bests[metric_name], metric_value)
                        metrics_compared[metric_name] += 1

            for method_name, method_results in method_dict.items():
                for metric_name, metric_value in method_results.items():
                    method_results[metric_name] = result_to_str(metric_value, "inf" in metric_name or "ASR" in metric_name)

                    if metric_value is not None and np.allclose(metric_value, metric_bests[metric_name]) \
                        and metrics_compared[metric_name] > 1:
                        method_results[metric_name] = rf"\textbf{{ {method_results[metric_name]} }}"

method_tex = {
    "cwk": r"CW^K",
    "ad": r"AD",
    "cvxproj": r"\textbf{QuadAttac$K$}"
}

doc = Document("multirow")

protocol_cols = 1
attack_method_cols = 1
best_cols = 4
mean_cols = 4
worst_cols = 4

if only_mean:
    col_widths = [protocol_cols, attack_method_cols, mean_cols]
else:
    col_widths = [protocol_cols, attack_method_cols, best_cols, mean_cols, worst_cols]

total_cols = sum(col_widths)
tabular_string = "|"

for w in col_widths:
    tabular_string += "l" * w + "|"

table1 = Tabular(tabular_string)
table1.add_hline()
table1.add_row((MultiColumn(total_cols, align='|c|', data=NoEscape(model_to_tex[model_name])),))
table1.add_hline()

if only_mean:
    table1.add_row((
        MultiRow(2, data="Protocol"),
        MultiRow(2, data="Attack Method"),
        MultiColumn(mean_cols, align="|c|", data="Mean"),
    ))
else:
    table1.add_row((
        MultiRow(2, data="Protocol"),
        MultiRow(2, data="Attack Method"),
        MultiColumn(best_cols, align="|c|", data="Best"),
        MultiColumn(mean_cols, align="|c|", data="Mean"),
        MultiColumn(worst_cols, align="|c|", data="Worst"),
    ))

table1.add_hline(start=protocol_cols + attack_method_cols + 1)

num_result_colums = 1 if only_mean else 3
table1.add_row("", 
               "",
               *(NoEscape(r"ASR$\uparrow$"),
               NoEscape(r"$\ell_1 \downarrow$"),
               NoEscape(r"$\ell_2 \downarrow$"),
               NoEscape(r"$\ell_{\infty} \downarrow$"))*num_result_colums
               )

table1.add_hline()

for top_k, bs_dict in results.items():
    
    total_results = 0
    # Count total results
    for _, iter_dict in bs_dict.items():
        for num_iter, method_dict in iter_dict.items():
            total_results += len(method_dict)

    top_k_latex_obj = MultiRow(total_results, data=f"Top-{top_k}")

    shown_topk_obj = False

    for bs_steps, iter_dict in bs_dict.items():
        for num_iter, method_dict in iter_dict.items():
            for method_name, method_results in method_dict.items():
                first_obj = top_k_latex_obj if not shown_topk_obj else ""
                shown_topk_obj = True

                row_results = []

                reduction_names = ["mean"] if only_mean else ["best", "mean", "worst"]
                for reduction in reduction_names:
                    for metric in ["ASR", "L1", "L2", "L_inf"]:
                        result_key = f"{metric}_{reduction}"
                        long_result = "inf" in metric
                        row_results.append(
                            NoEscape(method_results[result_key])
                            )

                table1.add_row(
                    first_obj,
                    NoEscape("$" + method_tex[method_name] +
                            f"_{{{bs_steps}x{num_iter}}}$"),
                    *row_results
                )

            table1.add_hline(start=protocol_cols + 1)

    table1.add_hline()

# doc.append(table1)

print(table1.dumps())