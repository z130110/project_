import numpy as np
import sys
import ast
import csv
import random
import time

class CsvReader(object):
    DATA_DIR = "/data/celebrus_extractions/"
    def __init__(self, data_dir, file_name=None, features_to_read=None, percent_to_read=1.0, counts_for_each_target=sys.maxsize):
        self.data_dir = data_dir
        self.file_name = file_name
        self.full_data_path = CsvReader.DATA_DIR + "/" + self.data_dir + "/" + self.file_name
        self.percent_to_read = percent_to_read
        self.features_to_read = features_to_read
        self.data_columns = self.get_columns()
        self.selected_data = self.initialize_selected_data()

        self.counts_for_each_target = counts_for_each_target
        self.counter = self.initialize_counter()

    def initialize_selected_data(self):
        selected_data = {}
        for col in self.features_to_read:
            selected_data[col] = []
        #print(selected_data)
        return selected_data

    def initialize_counter(max_pr_target):
        res = {}
        for i in range(0, 200):
            res[i] = 0
        return res

    def get_columns(self):
        with open(self.full_data_path, 'r') as open_file:
            first_line = open_file.readline()

            # Removing trailing comma, and newline symbols
            first_line = first_line[1:-1].split(",")
        return first_line

    def read_raw_csv(self):
        # Needed Indicies
        index = []
        target_index = None
        for col in self.features_to_read:
            index.append((col, self.data_columns.index(col)))
            if col == "target":
                target_index = self.data_columns.index(col) + 1

        with open(self.full_data_path, newline='') as f:
            csv_reader = csv.reader(f, delimiter=',', quotechar='"')
            next(csv_reader)
            row_count = 0
            for row in csv_reader:

                age = eval(row[target_index])
                if age not in self.counter:
                    continue
                if self.counter[age] >= self.counts_for_each_target:
                    continue
                if (self.percent_to_read <= random.random()):
                    continue
                for col, i in index:
                    try:
                        self.selected_data[col].append(eval(row[i + 1]))
                    except:
                        self.selected_data[col].append(eval('"' + row[i + 1] + '"'))
                self.counter[age] += 1
                row_count += 1

        print("Number of lines extracted from CsvReader: ", row_count)
        return self.selected_data

if __name__ == "__main__":
    print("Comment out below for test")
    """
    csv_reader = CsvReader("data4", "0.5_10_1.csv", features_to_read=["target","data", "combined_normalizedpagetimeviewactivetime","combined_eventtimestamp"], counts_for_each_target=200)
    #print(csv_reader.data_columns)
    csv_reader.read_raw_csv()
    print(csv_reader.selected_data["target"][:20])
    """
