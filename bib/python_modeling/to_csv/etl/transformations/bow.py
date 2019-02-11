import numpy as np
from sklearn import preprocessing
import time
import collections
import json

def flatten(x):
    out = []
    for i in x:
        for j in i:
            out.append(j)
    return np.array(out)

class BOW(object):
    def __init__(self, page_clicks, json_name = "latestrep"):
        self.lookup, self.distinct = self.create_one_hot_encoder(page_clicks)
        
        out_dict = {}
        for key in self.lookup:
            out_dict[key] = int(self.lookup[key])
        
        json_filename = json_name + ".json"
        with open('/home/Workdir/bib/python_modeling/to_csv/json/' + json_filename, 'w') as fp:
            json.dump(out_dict, fp)
        
        self.bow_rep = self.create_bow_rep(page_clicks)

    def create_one_hot_encoder(self, all_pagelocations):
        all_pages = flatten(all_pagelocations)
        distinct_pages = np.unique(all_pages)
        page_encoder = preprocessing.LabelEncoder()
        fitted_page_encoder = page_encoder.fit(distinct_pages)
        indices = fitted_page_encoder.transform(distinct_pages)
        lookup = dict(zip(list(distinct_pages), indices))
        return lookup, len(distinct_pages)
    
    def populate_single_bow(self, session_clicks):
        out_arr = np.zeros((self.distinct))
        for click in session_clicks:
            idx = self.lookup[click]
            out_arr[idx] += 1
        return out_arr
    
    def create_bow_rep(self, page_clicks):
        fin_arr = np.zeros([len(page_clicks), self.distinct])
        
        for i, click_session in enumerate(page_clicks):
            fin_arr[i,:] = self.populate_single_bow(click_session)
        return fin_arr
    
    def run(self):
        return self.bow_rep
            
    
     