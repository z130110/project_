import numpy as np
from sklearn import preprocessing
import time

class LoginTime(object):
    def __init__(self, hour, minute, day):
        self.transformed_time = self.transform_login_time(hour, minute)
        self.transformed_day = self.transform_login_day(day)        
        
    def transform_login_day(self, day):
        day = np.array(day)
        return np.cos((2*np.pi * day) / 6), np.sin((2*np.pi * day) / 6)
    
    def transform_login_time(self, hour, minute):
        hour = np.array(hour)
        minute = np.array(minute)
        comb = hour * 60 + minute
        return np.cos((2*np.pi * comb) / 1440), np.sin((2*np.pi * comb) / 1440)
    
    def run(self):
        return self.transformed_time[0],self.transformed_time[1], self.transformed_day[0],self.transformed_day[1]