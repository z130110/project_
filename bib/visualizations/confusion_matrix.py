import numpy as np
from matplotlib import pyplot as plt
#import pyspark.sql.functions as func
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import itertools

class confusion_matrix(object):
    def __init__(self, df=None, true_label="target", pred_label="prediction", np_label = None,np_pred = None, normalize = True,\
            title = "Confusion matrix, without normalization", cmap = plt.cm.Purples, bin_data=False, bin_size=5):
        self.df = df
        self.true_label = true_label
        self.pred_label = pred_label
        self.np_label = np_label
        self.np_pred = np_pred
        self.normalize = normalize
        self.title = title
        self.cmap = cmap
        self.conf_mat = None
        self.classes = None
        self.bin_data = bin_data
        if self.bin_data:
            self.np_pred, self.np_label, self.intervals = self.perform_bin_data(np_pred,np_label,bin_size)
        
    def df_to_conf_mat(self):
        # makes name of label and prediction uniformly
        df = self.df.withColumnRenamed(self.true_label,"target")
        df = df.withColumnRenamed(self.pred_label,"prediction")
        group_combination = df.groupby("target","prediction").agg(func.count("*").alias("count_target"))
        np_group_com = np.array(group_combination.collect())
        num_rows = np.array(df.groupby("target").agg(func.count(df.target).alias("count_target")).collect()).shape[0]
        temp_classes = np.array(df.groupby("target").agg(func.count(df.target).alias("count_target")).collect())[:,0]
        temp_classes.sort()
        classes = [str(i) for i in temp_classes]
        age_start = np_group_com[:,0].min()
        conf_mat = np.zeros([num_rows,num_rows])
        conf_mat[np_group_com[:,0] - age_start, np_group_com[:,1] - age_start] = np_group_com[:,2]
        self.conf_mat = conf_mat
        self.classes = classes
        return conf_mat, classes   
    
    def np_to_conf_mat(self):
        conf_mat = sklearn_confusion_matrix(self.np_label, self.np_pred)
        temp_classes = np.array(list(set(self.np_label)))
        temp_classes.sort()
        classes = [str(i) for i in temp_classes]
        self.conf_mat = conf_mat
        self.classes = classes
        if self.bin_data:
            self.classes = self.intervals
        return conf_mat, classes
    
    def plot_conf_mat(self):     
        if self.normalize:
            con_mat = self.conf_mat.astype('float') / self.conf_mat.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            con_mat = self.conf_mat
            print('Confusion matrix, without normalization')
        #print(con_mat)
        classes = self.classes        
        plt.imshow(con_mat, interpolation = "nearest", cmap = self.cmap)
        plt.title(self.title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')   
        plt.show()
        
    def create_interval(self,bin_size):
        out_bins = []
        cnt = 0
        for i in list(range(19,86)):
            cnt += 1
            if cnt == bin_size:
                out_bins.append(i)
                cnt = 0
        return out_bins

    def place_into_bin(self, pred, intervals):
        for i in range(len(intervals)):
            if pred < intervals[i]:
                return i
        return i

    def perform_bin_data(self, predictions, y, bin_size):
        intervals = self.create_interval(bin_size)
        pred_out = []
        y_out = []

        for i in range(len(predictions)):
            y_pred = predictions[i]
            y_true = y[i]
            pred_out.append(self.place_into_bin(y_pred, intervals))
            y_out.append(self.place_into_bin(y_true, intervals))
        return np.array(pred_out), np.array(y_out), intervals

    
    def set_classes(self,new_classes):
        self.classes = new_clases
