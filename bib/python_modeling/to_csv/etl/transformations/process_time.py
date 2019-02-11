import numpy as np
from sklearn import preprocessing
import time
from sklearn.decomposition import PCA

def flatten(x):
    out = []
    for i in x:
        for j in i:
            out.append(j)
    return np.array(out)

class ProcessTime(object):
    def __init__(self, page_location, event_timestamp, num_components):
        event_time = self.timestamp_to_time(event_timestamp)
        # create lookup table for [pagelocation => index] 
        self.lookup, self.num_classes = self.create_one_hot_encoder(page_location)
        self.normalized_accumulated_time =\
            self.accumulate_page_time(page_location,event_time)
        pca = PCA(n_components=num_components)
        self.pca = pca.fit_transform(self.normalized_accumulated_time)
        
    def accumulate_page_time(self,page_locations, page_time):
        out = np.zeros([len(page_locations),self.num_classes])
        for i in range(len(page_locations)):
            # for lists of pagelocation, find the indicies of the pagelocations and accumulate these
            index = self.get_category_index(page_locations[i])
            # get the existing time spent
            time_spend = page_time[i]
            # accumulate the index
            accumulated_time = self.accumulate_index(index,time_spend)
            normalized_accumulated_time = accumulated_time / len(page_locations[i])
            out[i,:] = normalized_accumulated_time
        return self.normalize_time_each_page(out)

    def create_one_hot_encoder(self, all_pagelocations):
        all_pages = flatten(all_pagelocations)
        distinct_pages = np.unique(all_pages)
        page_encoder = preprocessing.LabelEncoder()
        fitted_page_encoder = page_encoder.fit(distinct_pages)
        indices = fitted_page_encoder.transform(distinct_pages)
        lookup = dict(zip(distinct_pages, indices))
        return lookup, len(distinct_pages)

    def get_category_index(self,pagelocations):
        index_out = []
        for page in pagelocations:
            index_out.append(self.lookup[page])
        return index_out

    def accumulate_index(self,index,time_spend):
        acc = np.zeros([self.num_classes])
        for i in range(len(index)):
            acc[index[i]] += time_spend[i]
        return acc

    def timestamp_to_time(self,time):
        fin_array = []
        for time_list in time:
            out = []
            for i in range(0,len(time_list)-1):
                time = time_list[i+1] - time_list[i]
                out.append(time)
            out.append(0)
            fin_array.append(out)
        return np.array(fin_array)

    def normalize_time_each_page(self,accumulated_time):
        for i in range(accumulated_time.shape[1]):
            _sum = float(np.sum(accumulated_time[:,i]))
            if _sum != 0:
                accumulated_time[:,i] = accumulated_time[:,i] / _sum
        return accumulated_time

    def run(self):
        #return self.normalized_accumulated_time
        return self.pca
