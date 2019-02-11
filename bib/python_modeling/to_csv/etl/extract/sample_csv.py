from pyspark.sql.functions import udf, concat_ws

class SampleCSV(object):
    """
    Convert single column to feature
    @param
    """
    def __init__(self,df,FC):
        self.df = df
        self.FC = FC
    
    def sub_sample(self,df,pct):
        data,_ = self.df.randomSplit([pct, (1-pct)], 24)
        return data
    
    def save_csv(self,df,filename):
        filename  = "/data/celebrus_extractions/data5/" + filename
        p = df.toPandas()
        print("shape of save: ")
        print(p.shape)
        #df.toPandas().to_csv(filename)
        p.to_csv(filename)
        
    def save_mult(self,df,pct):
        sampled_df = self.sub_sample(df,pct)
        size_list = [10,15,35,80,120,240,380,500]
        wind_list = [1,2,3,4,5,6,7,8,9,10]
        
        
        for vec_len in size_list:
            for window_size in wind_list:
                
                print("vector length: {}".format(str(vec_len)))
                print("window size  : {}".format(str(window_size)))
                
                file_name = str(pct) + "_" + str(vec_len) + "_" + str(window_size) + ".csv" 
                formatted = self.FC.compute_word2vec(sampled_df,vec_len,window_size,sub_test=True)
                print("saving")
                self.save_csv(formatted,file_name)