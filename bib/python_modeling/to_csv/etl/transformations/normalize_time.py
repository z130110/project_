import numpy as np

class NormalizeTime(object):
    def __init__(self, event_timestamp, cut_off=40):
        self.event_timestamp = event_timestamp
        self.cut_off = cut_off

    def zero_pad_time(self):
        pad_time_stamp = []
        for data in self.event_timestamp:
            if len(data) >= self.cut_off:
                pad_time_stamp.append(data[:self.cut_off])
            else:
                pad_time_stamp.append(data + ([0] * (self.cut_off - len(data))))
        return pad_time_stamp

    def normalize_zero_padded_time(self, X):
        X_time = np.array(X).astype(float)
        tmp_X = np.zeros((X_time.shape[0], X_time.shape[1] + 1))
        tmp_X[:,1:] = X_time
        tmp_X = tmp_X[:,:-1]
        X_r = X_time - tmp_X
        X_r[:,0] = 0
        X_r[np.where(X_r < 0)] = 0
        X = X_r

        res = np.array(X_r)

        for i in range(1, self.cut_off):
            #print((X[:,col] - X[:,col].mean()) / X[:,col].std())
            if np.sum(res[:,i]) > 0:
                res[:, i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
            else:
                continue
        res[:,0] = 0

        return res

    def run(self):
        zero_padded_t = self.zero_pad_time()
        normalized_time = self.normalize_zero_padded_time(zero_padded_t)
        return normalized_time
