from read_csv import CsvReader
from process_time import ProcessTime
import sys
sys.path.insert(0, "/home/Workdir/bib/constants")
import directories as DIR
import numpy as np
from sklearn.decomposition import PCA

def convert_target_to_class(targets):
    new_targets = np.empty_like(targets)
    for i in range(len(targets)):
        new_targets[i] = targets[i] < 28
    return new_targets

def convert_data(train):
    a = np.zeros([len(train),len(train[0])])
    for i in range(len(train)):
        a[i,:] = np.asarray(train[i])
    return a

def split_data(x,y,num_test):
    x_test  = x[:num_test]
    x_train = x[num_test:]
    y_test  = y[:num_test]
    y_train = y[num_test:]
    print("initial size: {}".format(y.shape))
    print("train       : {}".format(y_train.shape))
    print("test        : {}".format(y_test.shape))
    print("\n")
    print(x_test[0])
    print("\n")
    print(x_train[0])
    print("\n")
    return x_train, x_test, y_train, y_test
    
class FeatureModel(object):
    def __init__(self, model_description, x, y, num_test):
        self.model_description = model_description
        self.x_train, self.x_test, self.y_train, self.y_test = split_data(x, y, num_test)

class FeatureSelector(object):
    def __init__(self, folder, filename, data):
        self.data = data
        self.processed_time = ProcessTime(self.data["page"], self.data["time"]).run()
        self.processed_time_active = ProcessTime(self.data["page"], self.data["viewtime"]).run()
        num_rows = self.data["data"].shape[0]
        self.num_test =  int(num_rows // 10)
        self.models = self.initialize_models()
        
    def initialize_models(self):
        #model1 = self.create_model_1()
        #model2 = self.create_model_2()
        #model3 = self.create_model_3()
        #model4 = self.create_model_4()
        model5 = self.create_model_5()
        #model7 = self.create_model_7()

        #model6 = self.create_model_6()

        #return [model1, model2, model3, model4]
        #return  [model2, model3, model4]
        #return [model4]
        return [model5]
        
    def get_models(self):
        return self.models
    
    def create_model_1(self):
        """
        x : word2vec of given vector and window size
        y : age targets
        """
        x    = self.data["data"]
        y    = self.data["target"]
        description = "model1" 
        return FeatureModel(description, x, y, self.num_test)

    def create_model_2(self):
        """
        x : normalized processed time
        y : age targets
        """
        x     = self.processed_time
        y     = self.data["target"]
        description  = "model2"
        return FeatureModel(description, x, y, self.num_test)

    def create_model_3(self):
        """
        x : clicks + normalized processed time
        y : age targets
        """
        clicks = self.data["data"]
        time   = self.processed_time
        x      = np.concatenate((clicks,time), axis=1)
        y      = self.data["target"]
        description = "model3"
        return FeatureModel(description, x, y, self.num_test)
    
    def create_model_4(self):
        """
        x : clicks + time-pca
        y : age targets
        """
        pca       = PCA(n_components=4)
        clicks    = self.data["data"]
        time      = self.processed_time
        pca.fit(time)
        pca_time = pca.transform(time)
        x        = np.concatenate((clicks,pca_time), axis=1)
        y        = self.data["target"]
        description = "model4"
        return FeatureModel(description, x, y, self.num_test)
    
    def create_model_5(self):
        """
        test
        """
        clicks = self.data["data"]
        view_time = self.processed_time_active
        browser = self.data["browser"]
        os = self.data["os"]
        hour = self.data["hour"]
        day = self.data["weekday"]
        y = self.data["target"]
        #print("target")
        #print(y)
        description = "model5"
        x = np.column_stack((clicks, view_time, browser, os, day, hour))
        return FeatureModel(description, x, y, self.num_test)
    
    def create_model_6(self):
        """
        test
        """
        clicks = self.data["data"]
        browser = self.data["browser"]
        os = self.data["os"]
        hour = self.data["hour"]
        day = self.data["weekday"]
        y = self.data["target"]
        description = "model6"
        x = np.column_stack((clicks, browser, os, day, hour))
        return FeatureModel(description, x, y, self.num_test)
    

    def create_model_7(self):
        """
        test
        """
        clicks = self.data["data"]
        view_time = self.processed_time_active
        browser = self.data["browser"]
        os = self.data["os"]
        hour = self.data["hour"]
        day = self.data["weekday"]
        y = self.data["target"]
        description = "model7"
        x = np.column_stack((view_time, browser, os, day, hour))
        return FeatureModel(description, x, y, self.num_test)

if __name__ == '__main__':
    data = CsvReader("data1", "0.5_10_5.csv").run()
    print("page")
    print((data["page"][5]))
    print(len(data["page"][5]))
    print("timestamps")
    print((data["time"][5]))
    print(len(data["time"][5]))
    print("\n")
    b = FeatureSelector("data1","0.5_10_5.csv")
    #print(b.processed_time)
    #print(b.get_models())