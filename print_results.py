from modelguidedattacks import results

results_list = results.load_all_results()

filter = {
    "loss": results.in_set(["cwk"]),
    "model": results.eq("vit_base"),
    "k": results.eq(5),
    "binary_search_steps": results.eq(1),
    "unguided_iterations": results.eq(30),
    # "topk_loss_coef_upper": results.eq(20),
    # "unguided_lr": results.eq(0.002),
    "cvx_proj_margin": results.eq(0.2),
    # "seed": results.eq(10),
}

filtered_results = results.filter_results(filter, results_list)

print ("Found", len(filtered_results))

for result in filtered_results:
    print ("-" * 30)
    for key, val in result.items():
        print (key, "=", val)
    print ("-" * 30)
