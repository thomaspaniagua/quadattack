import numpy as np
import sys
from modelguidedattacks import results

results_list = results.load_all_results()

filter = {
    "loss": results.in_set(["cvxproj"]),
    "model": results.eq("resnet50"),
    "k": results.eq(20),
    "binary_search_steps": results.eq(1),
    "unguided_iterations": results.eq(60),
    # "topk_loss_coef_upper": results.eq(20),
    # "unguided_lr": results.eq(0.002),
    "cvx_proj_margin": results.eq(0.2),
    "topk_loss_coef_upper": results.gte(12)
    # "seed": results.eq(10),
}

filtered_results = results.filter_results(filter, results_list)

combined_results = {}

for result in filtered_results:
    for key in result:
        if key not in combined_results:
            combined_results[key] = []

        combined_results[key].append(result[key])

unique_runs = len(np.unique(combined_results["seed"]))
print ("Stats from", len(filtered_results))
# assert len(combined_results["seed"]) == unique_runs

for key, val in list(combined_results.items()):
    if key in ["ASR", "L1", "L2", "L_inf"]:
        val = np.array(val)
        combined_results[f"{key}_mean"] = np.mean(val[np.isfinite(val)])
        combined_results[f"{key}_median"] = np.median(val[np.isfinite(val)])

# Coupled results
best_asr_idx = np.argmax(combined_results["ASR"])
best_asr = combined_results["ASR"][best_asr_idx]
best_l1 = combined_results["L1"][best_asr_idx]
best_l2 = combined_results["L2"][best_asr_idx]
best_linf = combined_results["L_inf"][best_asr_idx]

combined_results["ASR_best"] = best_asr
combined_results["L1_best"] = best_l1
combined_results["L2_best"] = best_l2
combined_results["L_inf_best"] = best_linf

worst_asr_idx = np.argmin(combined_results["ASR"])
worst_asr = combined_results["ASR"][worst_asr_idx]
worst_l1 = combined_results["L1"][worst_asr_idx]
worst_l2 = combined_results["L2"][worst_asr_idx]
worst_linf = combined_results["L_inf"][worst_asr_idx]

combined_results["ASR_worst"] = worst_asr
combined_results["L1_worst"] = worst_l1
combined_results["L2_worst"] = worst_l2
combined_results["L_inf_worst"] = worst_linf

draw_keys = ["best", "mean", "worst"]
val_keys = ["ASR", "L1", "L2", "L_inf"]

print ("---------------")
for draw_key in draw_keys:
    for val_key in val_keys:
        key = val_key + "_" + draw_key
        val = combined_results[key]

        print (key, val)
print ("---------------")

for draw_key in draw_keys:
    for val_key in val_keys:
        key = val_key + "_" + draw_key
        val = combined_results[key]

        if np.isinf(val):
            val = "N/A"

        sep = "&"
        if isinstance(val, str):
            sys.stdout.write(f"{val} {sep} ")    
        elif "inf" in key:
            sys.stdout.write(f"{val:.3f} {sep} ")
        else:
            sys.stdout.write(f"{val:.2f} {sep} ")