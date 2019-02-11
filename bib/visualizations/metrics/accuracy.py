import random, string
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
#from keras.utils import plot_model


class PlotCustomAccuracy(object):
    def __init__(self, accuracy, train_val, test_val, data_dir, prefix="", file_name = ""):
        self.acc = accuracy
        self.train_val = train_val
        self.test_val = test_val
        self.epochs = list(range(1, len(accuracy) + 1))
        print(self.epochs)
        self.file_name = file_name
        if file_name == "":
            self.file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        self.save_dir_plot = "/data/python_results/" + data_dir + "/" + \
            prefix +  "performance_" + self.file_name  + ".png"
        print(self.save_dir_plot)
        #self.save_dir_keras_plot = "/data/python_results/" + data_dir + "/" + \
        #    prefix +  "model_" + self.file_name  + ".png"

    def plot(self):
        plt.plot(self.epochs, self.acc, 'bo', label='Training loss')
        plt.plot(self.epochs, self.train_val, 'b', label='Validation train loss')
        plt.plot(self.epochs, self.test_val, 'bo', label='Validation test loss', color='red')
        plt.title('Training and validation accuracy')
        plt.legend()

        """
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        """

        plt.axhline(y = 0.3, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.4, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.5, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.6, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.7, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.8, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.axhline(y = 0.9, linewidth=1, linestyle='dashed', color='black', alpha = 0.5)
        plt.ylim((0.0,1.00))

        #print(self.save_dir_plot)
        #print(self.save_dir_keras_plot)
        plt.savefig(filename = self.save_dir_plot)
        plt.clf()

    #def plot_keras_model(self, keras_model):
    #    plot_model(keras_model, to_file=self.save_dir_keras_plot, show_shapes=True)
    #    return self
