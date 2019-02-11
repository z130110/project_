from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, expr, when, lower, collect_set,collect_list
from pyspark.ml.feature import StringIndexer
from fc_get_starttime import GetStartTime
from functools import reduce
import pyspark.sql.functions as func

def convert_now(x):
    return list(map(str,x))

def cut_pages_for_sessions(clicks):
    out = []
    for click in clicks:
        if "danskebank.dk/privat/bliv-kunde" in click:
            if out == []:
                return None
            return out
        else:
            out.append(click)
    return out

def netbank(clicks):
    for click in clicks:
        if "netbank2" in click:
            return int(1)
    return int(0)

class FC(object):
    """
    Convert single column to feature
    @param
    """
    def __init__(self,spark, df, input_col="combined_pagelocation", output_col="data"):
        self.spark      = spark
        self.df         = df.na.drop()
        self.input_col  = input_col
        self.output_col = output_col
        
    def group_column_by_session(self):
        """
        Group a single column, for each sessionnumber
        @param collected_column : this is the name of the column to merge into seperate lists, the output column will have the same name
        @return the dataframe only with the specified column merged into a list, for each sessionnumber
        """
        """
        # aggregate pagelocation
        df_page_collected = self.df.groupBy("sessionnumber")\
                  .agg(collect_list("pagelocation").alias("combined_pagelocation"))
        
        # aggregate eventtimestamps
        df_time_collected = self.df.groupBy("sessionnumber")\
            .agg(collect_list("eventtimestamp").alias("combined_eventtimestamp")).distinct()
        
        # retrieve sessionstarttime, weekday and hour
        df_time_collected = GetStartTime(df_time_collected).run()
        
        # retrieve active pageviewtime
        df_pageviewactive_collected = self.df.groupBy("sessionnumber")\
            .agg(collect_list("pageviewactivetime").alias("combined_pageviewactivetime")).distinct()
            
        df_scrolldistance_collected = self.df.groupBy("sessionnumber")\
            .agg(func.avg("pagescrollmaxdistance").alias("average_scrolldistance"))
            
        #print("combining")
        
        df_norm_pageviewactive_collected = self.df.groupBy("sessionnumber")\
            .agg(collect_list("normalized_pagetimeviewactivetime").alias("combined_normalizedpagetimeviewactivetime")).distinct()
    
        """

        """
        #TOOO GET BACK AFTER
        df_collected = self.df.groupBy("sessionnumber")\
            .agg( collect_list("pagelocation").alias("combined_pagelocation"),            \
                  func.avg("pageviewactivetime").alias("average_pageview"),               \
                  collect_list("eventtimestamp").alias("combined_eventtimestamp"),        \
                  collect_list("pageviewactivetime").alias("combined_pageviewactivetime"),\
                  collect_list("normalized_pagetimeviewactivetime").alias("combined_normalizedpagetimeviewactivetime"), \
                  func.avg("pagescrollmaxdistance").alias("average_scrolldistance")).distinct()

        """
        
        cut_page_udf = udf(cut_pages_for_sessions,ArrayType(StringType()))
        #netbank_udf = udf(netbank)
        
        df_collected = self.df.groupBy("sessionnumber")\
            .agg( collect_list("pagelocation").alias("combined_pagelocation"),            \
                  collect_list("eventtimestamp").alias("combined_eventtimestamp")).distinct()
        #df_collected = df_collected.withColumn("netbank", netbank_udf(df_collected['combined_pagelocation']))
        #df_collected = df_collected.where(df_collected.netbank == "0")
        #df_collected = df_collected.withColumn("combined_pagelocation2", cut_page_udf(df_collected['combined_pagelocation']))
        #df_collected = df_collected.where(col("combined_pagelocation2").isNotNull())
        # retrieve normalized pageviewtime
        
        # join all tables to create features
        """
        df_page_collected_joined = df_page_collected.join(self.df, "sessionnumber", "inner")\
            .drop(*["aftlnr","pagelocation","eventtimestamp"])\
            .join(df_pageviewactive_collected, "sessionnumber", "inner")\
            .drop(*["pageviewactivetime", "normalized_pagetimeviewactivetime"])\
            .join(df_norm_pageviewactive_collected, "sessionnumber", "inner")\
            .join(df_time_collected, "sessionnumber", "inner").distinct()
        """
        
        df_time_collected = GetStartTime(df_collected).run()
        
        """
        df_page_collected_joined_new = df_collected.join(self.df, "sessionnumber", "inner")\
            .join(df_time_collected, "sessionnumber", "inner")\
            .drop(*["aftlnr","pagelocation","eventtimestamp","pageviewactivetime", "normalized_pagetimeviewactivetime",\
                    "pagescrollmaxdistance"]).distinct()
         #.join(df_time_collected, "sessionnumber", "inner")\
        """   
        df_page_collected_joined_new = df_time_collected.join(self.df, "sessionnumber", "inner")\
            .drop(*["aftlnr","pagelocation","eventtimestamp","pageinstanceid"]).distinct()
        
        return df_page_collected_joined_new    
        
    def compute_word2vec(self, input_df, output_vec_len, window_size=5,sub_test=False):
        """
        Compute the word2vec for a given dataframe
        @param input_df       : the dataframe to perform the action upon
        @param output_vec_len : the length (int) of the output vector
        @param input_col      : the name (string) of the input column
        @param output_col     : the name (string) of the output column
        @return output dataframe with output column
        """
        # ensure that the input column is of type StringType()
        toArray = udf(lambda vs: vs, ArrayType(StringType()))
        toArray1= udf(lambda vs: vs.toArray())
        df = input_df.withColumn(self.input_col, toArray(input_df[self.input_col]))
        # initialize word2vec
        word2Vec = Word2Vec(vectorSize=output_vec_len, windowSize=window_size,minCount=5,inputCol=self.input_col, outputCol=self.output_col)
        # train word2vec model
        model = word2Vec.fit(df)
        # compute transformation
        result = model.transform(df)
        # convert result to a vector
        if not sub_test:
            conv = udf(lambda vs: Vectors.dense(vs), VectorUDT())
            out = result.withColumn(output_col, conv(result[output_col]))
            return out
        else:
            return result