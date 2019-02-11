import numpy as np
import collections

class ConvertClass(object):
    def __init__(self, targets):
        targets = np.array(targets)
        self.converted_targets = self.convert_to_class(targets)
 
    def convert_to_class(self, targets):
        targets[np.where(targets <= 27)] = 1
        targets[np.where(targets > 27)]  = 0
        return targets
        
    def run(self):
        return self.converted_targets