import numpy as np
import collections

def get_lowest_val(d):
    return min(d.values())

def convert_target_to_class(targets):
    new_targets = np.empty_like(targets)
    for i in range(len(targets)):
        new_targets[i] = int(targets[i] < 28)
    return new_targets

class BalanceDataset(object):
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
        upper = np.argwhere(sorted_array[:,self.target_col] > 18)[0][0]
        lower = np.argwhere(sorted_array[:,self.target_col] == 85)[0][0]
        sorted_array = sorted_array[upper:lower,:]
        #if self.balance == "classification":
        #    sorted_array[:,self.target_col] = convert_target_to_class(sorted_array[:,self.target_col])
        #print("here",sorted_array)
        #print(np.unique(sorted_array[:,-1], return_counts=True))
        ages = np.unique(sorted_array[:,self.target_col])
        lowest_freq = get_lowest_val(collections.Counter(sorted_array[:,self.target_col]))# // 4
        out = None
        for age in ages:
            age_group = sorted_array[np.where(sorted_array[:,self.target_col] == age)]
            sample = age_group[np.random.choice(age_group.shape[0], lowest_freq, replace=False), :]
            if out is None:
                out = sample
            else:
                out = np.vstack((out, sample))

        return out

    def run(self):
        return self.balanced
