import numpy as np

class TimeDiscretize(object):
    def __init__(self, event_timestamp):
        self.event_timestamp = event_timestamp

    def zero_pad_time(self):
        pad_time_stamp = []
        for data in self.event_timestamp:
            if len(data) >= 40:
                pad_time_stamp.append(data[:40])
            else:
                pad_time_stamp.append(data + ([0] * (40 - len(data))))
        return pad_time_stamp

    def discretize_time(self, X):
        X_time = np.array(X).astype(float)
        tmp_X = np.zeros((X_time.shape[0], X_time.shape[1] + 1))
        tmp_X[:,1:] = X_time
        tmp_X = tmp_X[:,:-1]
        X_r = X_time - tmp_X
        X_r[:,0] = 0
        X_r[np.where(X_r < 0)] = 0
        X = X_r

        res = np.array(X_r)

        res = np.round(res / 1000).astype(int)
        #print(res[:10,:10])
        return res

    def run(self):
        zero_padded_t = self.zero_pad_time()
        res = self.discretize_time(zero_padded_t)
        return res
