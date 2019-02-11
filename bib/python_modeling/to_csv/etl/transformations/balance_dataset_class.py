import numpy as np
import collections

def get_lowest_val(d):
    return min(d.values())

class BalanceDatasetClass(object):
    """
    Convert single column to feature
    @param
    """
    def __init__(self, dataset, col_idx):
        self.col_idx = col_idx
        self.target_col = col_idx["target"]
        self.balanced = self.remove_below_n_sessions(dataset)

    def remove_below_n_sessions(self, dataset):
        # sort array by target
        sorted_array = dataset[dataset[:,self.target_col].argsort()]
        ages = np.unique(sorted_array[:,self.target_col])
        lowest_freq = get_lowest_val(collections.Counter(sorted_array[:,self.target_col]))# // 4
        out = None
        for age in ages:
            age_group = sorted_array[np.where(sorted_array[:,self.target_col] == age)]
            print(age)
            print(age_group.shape)
            sample = age_group[np.random.choice(age_group.shape[0], lowest_freq, replace=False), :]
            if out is None:
                out = sample
            else:
                out = np.vstack((out, sample))
        return out

    def run(self):
        return self.balanced
