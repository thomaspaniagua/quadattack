import os
import torch
from pathlib import Path
import numpy as np
import copy
from collections import OrderedDict

config_parameter_keys = ["loss", "unguided_lr", "model", "k", "binary_search_steps",
                        "unguided_iterations", "topk_loss_coef_upper", "seed",
                        "opt_warmup_its", "cvx_proj_margin", 
                        "topk_loss_coef_upper", "binary_search_steps"]

def config_to_dict(config):
    result_keys = config_parameter_keys
        
    result_dict = {
        key: getattr(config, key) for key in result_keys
    }

    if hasattr(config, "cvx_proj_margin"):
        result_dict["cvx_proj_margin"] = config.cvx_proj_margin
    else:
        result_dict["cvx_proj_margin"] = 0.2

    return result_dict

def load_all_results(load_min_max=False):
    if not os.path.isdir("results_rebuttal"):
        return []
    
    all_result_files = Path('results_rebuttal').rglob('*.save')

    results_list = []
    for result_file in all_result_files:
        result = torch.load(result_file)
        config = result["config"]
        
        result_dict = config_to_dict(config)

        result_dict["ASR"] = result["ASR"]
        result_dict["L1"] = result["L1 Energy"]
        result_dict["L2"] = result["L2 Energy"]
        result_dict["L_inf"] = result["L_inf Energy"]

        if "L2 Energy Max" in result and load_min_max:
            result_dict["L1 Max"] = result["L1 Energy Max"]
            result_dict["L2 Max"] = result["L2 Energy Max"]
            result_dict["L_inf Max"] = result["L_inf Energy Max"]

            result_dict["L1 Min"] = result["L1 Energy Min"]
            result_dict["L2 Min"] = result["L2 Energy Min"]
            result_dict["L_inf Min"] = result["L_inf Energy Min"]

        results_list.append(result_dict)

    return results_list

def close(target, eps=1e-5):
    return lambda x: np.allclose(x, target, atol=eps)

def eq(target):
    if isinstance(target, float):
        return close(target)
    else:
        return lambda x: x == target

def gte(target):
    return lambda x: float(x) >= target

def lte(target):
    return lambda x: float(x) <= target

def in_set(target):
    return lambda x: x in target

def filter_from_config(config):
    config_dict = config_to_dict(config)

    filter = {
        key: eq(val) for (key, val) in config_dict.items()
    }

    return filter

def filter_results(filter, results_list, only_with_minmax=False):
    filtered_results = []
    for result in results_list:
        pass_filter = True
        for key, val in result.items():
            if key not in filter:
                continue

            if not filter[key](val):
                pass_filter = False
                break

        if only_with_minmax and "L2 Max" not in result:
            continue

        if pass_filter:
            filtered_results.append(result)

    return filtered_results

def resolve_nonunique_filter(filter, results_list, include_failed=False):
    filtered_results = filter_results(filter, results_list)

    unique_parameters = []
    # Find unique parameter sets for results
    for result in filtered_results:
        result_parameters = {param_key:result[param_key] for param_key in config_parameter_keys}

        # Round to avoid floating pt imprecision from messing with set uniqueness checks
        for key in result_parameters.keys():
            if isinstance(result_parameters[key], float):
                result_parameters[key] = round(result_parameters[key], 5)

        del result_parameters["seed"]
        unique_parameters.append(result_parameters)

    # Only keep unique dicts
    unique_parameters = [dict(y) for y in set(tuple(x.items()) for x in unique_parameters)]

    best_metric = -np.Infinity
    best_param_set = None
    best_result_list = None
    for param_set in unique_parameters:
        # Perform another search
        unique_filter = {
            param_name: eq(param_value) for param_name, param_value in param_set.items()
        }

        filtered_results = filter_results(unique_filter, results_list)

        assert len(filtered_results) == 5

        asrs = [result["ASR"] for result in filtered_results]
        l2_energies = [result["L2"] for result in filtered_results]

        mean_asr = np.mean(np.array(asrs)[np.isfinite(asrs)])
        mean_l2 = np.mean(np.array(l2_energies)[np.isfinite(l2_energies)])

        # Arbitrary point in tradeoff curve
        result_goodness = -mean_l2 + mean_asr * 100

        if (mean_asr > 0 and mean_asr < 0.025) and not include_failed:
            # Irrelevant result and associated energies
            continue

        if result_goodness > best_metric or (include_failed and best_param_set is None):
            best_param_set = param_set
            best_result_list = filtered_results
            best_metric = result_goodness

    return best_param_set, best_result_list

def get_combined_results(filtered_results):
    combined_results = {}

    for result in filtered_results:
        for key in result:
            if key not in combined_results:
                combined_results[key] = []

            combined_results[key].append(result[key])

    unique_runs = len(np.unique(combined_results["seed"]))
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

    return combined_results

def build_full_results_dict(model_name="resnet50", verbose=False,
                            all_k=[20, 15, 10, 5, 1], 
                            all_num_iter=[60, 30],
                            all_search_steps=[1, 9],
                            all_methods=["cwk", "ad", "cvxproj"]):

    if verbose:
        print ("-" * 100)
        print ("Results for", model_name)

    results_list = load_all_results()
    results = OrderedDict()

    for k in all_k:
        results[k] = OrderedDict()

        for num_binary_search_steps in all_search_steps:
            results[k][num_binary_search_steps] = OrderedDict()

            for num_iter in all_num_iter:
                results[k][num_binary_search_steps][num_iter] = OrderedDict()

                for method_name in all_methods:
                    filter = {
                        "loss": eq(method_name),
                        "model": eq(model_name),
                        "k": eq(k),
                        "unguided_iterations": eq(num_iter),
                        "binary_search_steps": eq(num_binary_search_steps)
                    }

                    best_param_set, filtered_results = resolve_nonunique_filter(filter, results_list)

                    if verbose and best_param_set is not None:
                        print (f"K={k} Lr={best_param_set['unguided_lr']} and loss_coef={best_param_set['topk_loss_coef_upper']} ")
                
                    if best_param_set is None:
                        continue

                    assert len(filtered_results) == 5
                    
                    combined_results = get_combined_results(filtered_results)

                    for key in list(combined_results):
                        if "L1" not in key and "L2" not in key and "L_inf" not in key and "ASR" not in key:
                            del combined_results[key]

                    for key in list(combined_results):
                        if "mean" not in key and "worst" not in key and "best" not in key:
                            del combined_results[key]

                    results[k][num_binary_search_steps][num_iter][method_name] = combined_results

    return results

if __name__ == "__main__":
    build_full_results_dict(model_name="resnet50", verbose=True, all_search_steps=[1], all_methods=["cvxproj"])
    build_full_results_dict(model_name="densenet121", verbose=True, all_search_steps=[1], all_methods=["cvxproj"])
    build_full_results_dict(model_name="deit_small", verbose=True, all_search_steps=[1], all_methods=["cvxproj"])
    build_full_results_dict(model_name="vit_base", verbose=True, all_search_steps=[1], all_methods=["cvxproj"])
    x = 5