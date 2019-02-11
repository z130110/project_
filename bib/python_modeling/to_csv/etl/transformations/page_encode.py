import numpy as np
from sklearn import preprocessing
from itertools import groupby

def flatten(x):
    out = []
    for i in x:
        for j in i:
            out.append(j)
    return np.array(out)

class PageEncode(object):
    def __init__(self, page_location, cut_off=40):
        self.page_location = page_location
        self.cut_off=cut_off

    def one_hot_encoding(self):
        to_encode = preprocessing.LabelEncoder()
        to_encode.fit([p for data_point in self.page_location for p in data_point])
        print(to_encode)
        self.to_encode = to_encode
        largest_sentence = self.cut_off

        data_points = []
        for sentence in self.page_location:
            encoded_data = to_encode.transform(sentence)
            tmp = [x[0] + 1 for x in groupby(encoded_data)]
            tmp.extend([0] * (largest_sentence - len(tmp)))
            data_points.append(tmp[:self.cut_off])

        return data_points

    def to_2D_tensor(self, data_points):
        largest_sentence = len(data_points[0])
        #nr_words = int(max(list(map(max, data_points)))) + 1
        #res = np.zeros((len(data_points), largest_sentence, nr_words)).astype(int)

        """
        for i, data_p in enumerate(data_points):
            data_p = list(map(int, data_p))
            res[i,list(range(len(data_p))), data_p] = 1
        # Set first column to 0
        res[:,:,0] = 0
        """
        return res

    def run(self):
        return self.one_hot_encoding()
