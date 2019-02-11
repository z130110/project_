import numpy as np
from sklearn import preprocessing
import time
from scipy import stats

class GetAvgTime(object):
    def __init__(self, event_timestamp):
        self.normalized_time = self.get_avgtime(event_timestamp)
        
    def get_avgtime(self,eventtimestamp):
        fin_array = []
        std_array = []
        for time_list in eventtimestamp:
            out = []
            for i in range(0,len(time_list)-1):
                time = (time_list[i+1] - time_list[i]) / 1000
                if time > 200:
                    continue
                out.append(time)
            std_dev = np.std(out)
            average = np.average(out)
            if np.isnan(std_dev):
                std_array.append(0)
            else:
                std_array.append(std_dev)
            if average == 0 or np.isnan(average):
                fin_array.append(0)
            else:
                fin_array.append(average)
        fin_array = stats.zscore(np.array(fin_array))
        std_array = stats.zscore(np.array(std_array))
        
        #fin_array = np.array(fin_array)
        #std_array = np.array(std_array)
        return fin_array, std_array
    
    def run(self):
        return self.normalized_time
            
