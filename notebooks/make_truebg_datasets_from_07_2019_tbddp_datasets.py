import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

all_stats_df = pd.DataFrame()

# 476M BG datapoints
bg_array = np.load("/Users/ed/projects/predict-future-bgs/bg_data.npy")

number_snippets_in_chain = 15
n_datasets = 10
bg_array_len = len(bg_array)
dataset_start = [start_idx for start_idx in range(0, bg_array_len, int(np.floor(bg_array_len / n_datasets)))]
dataset_end = list(np.append(dataset_start[1:], bg_array_len))

for dataset_id, start_idx in enumerate(dataset_start[:-1]):
    end_idx = dataset_end[dataset_id]

    bg_subset = bg_array[start_idx:end_idx]
    bg_subset_rates = np.append([0], (bg_subset[1:] - bg_subset[:-1]) / 5)

    start_end_snippet = np.abs(bg_subset_rates) > 5
    snippet_indices = [i for i, val in enumerate(start_end_snippet) if val]

    start_indices = np.append([0], snippet_indices)
    end_indices = np.append(np.array(snippet_indices) - 1, [len(start_end_snippet)])
    snippet_length = end_indices - start_indices

    is_seven_to_ten_day_snippet = (snippet_length > 2016) & (snippet_length < 2880)
    seven_to_ten_day_snippets_start = start_indices[is_seven_to_ten_day_snippet]
    seven_to_ten_day_snippets_end = end_indices[is_seven_to_ten_day_snippet]
    seven_to_ten_day_snippets_lengths = snippet_length[is_seven_to_ten_day_snippet]

    snippet_df = pd.DataFrame(seven_to_ten_day_snippets_start, columns=["start"])
    snippet_df["end"] = seven_to_ten_day_snippets_end
    snippet_df["length"] = seven_to_ten_day_snippets_lengths
    snippet_df["val_at_start"] = bg_subset[snippet_df["start"].values]
    snippet_df["val_at_end"] = bg_subset[snippet_df["end"].values]

    current_snippet = 0
    snippet_order = np.array([current_snippet])
    temp_snippets = snippet_df.copy()

    snippet_bg_chain = bg_subset[snippet_df.loc[current_snippet, "start"] : snippet_df.loc[current_snippet, "end"]]

    for n in range(number_snippets_in_chain):
        print("chaining snippet {} of {}".format(n + 1, number_snippets_in_chain))
        temp_snippets.drop(index=current_snippet, inplace=True)
        temp_snippets["diff"] = np.abs(temp_snippets["val_at_start"] - snippet_df.loc[current_snippet, "val_at_end"])
        temp_snippets.sort_values(by="diff", inplace=True)
        print(np.abs(temp_snippets.iloc[0]["diff"] / 5))
        if np.abs(temp_snippets.iloc[0]["diff"] / 5) <= 2:
            next_snippet = temp_snippets.iloc[0].name
            snippet_order = np.append(snippet_order, next_snippet)
            current_snippet = copy.deepcopy(next_snippet)
            snippet_bg_chain = np.append(
                snippet_bg_chain,
                bg_subset[snippet_df.loc[current_snippet, "start"] : snippet_df.loc[current_snippet, "end"]],
            )
        else:
            break

    snippet_bg_chain_rates = (snippet_bg_chain[1:] - snippet_bg_chain[:-1]) / 5

    # capture snippet chain stats
    bg_df = pd.DataFrame(snippet_bg_chain, columns=["bg"])
    bg_df.to_csv(os.path.join("..", "data", "tbddp_bg_chains", "bg_{}.csv".format(dataset_id)))
    bg_stats_df = pd.DataFrame(bg_df.describe())
    rate_stats_df = pd.DataFrame(pd.DataFrame(snippet_bg_chain_rates, columns=["rate"]).describe())

    stats_df = pd.concat(
        [
            bg_stats_df.T.add_prefix("bg.").reset_index(drop=True),
            rate_stats_df.T.add_prefix("rate.").reset_index(drop=True),
        ], axis=1
    )

    all_stats_df = pd.concat([all_stats_df, stats_df], ignore_index=True)

    plt.hist(snippet_bg_chain)
    plt.show()

all_stats_df.to_csv(os.path.join("..", "data", "tbddp_bg_chains_stats.csv"))