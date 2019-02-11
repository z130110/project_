import numpy as np
from sklearn import preprocessing

class ProcessUserAgents(object):
    def __init__(self, os, browser):
        self.browser_indexed = self.one_hot_encode(browser)
        self.os_indexed = self.one_hot_encode(os)

    def one_hot_encode(self, useragent):
        distinct_agents = np.unique(useragent)
        agent_encoder = preprocessing.LabelEncoder()
        fitted_encoder = agent_encoder.fit(distinct_agents)
        return fitted_encoder.transform(useragent)

    def run(self):
        return self.os_indexed, self.browser_indexed,

    def run_browser(self):
        return self.browser_indexed

    def run_os(self):
        return self.os_indexed
