from pyspark.sql.functions import col, regexp_replace, lower
import pyspark.sql.functions as func

class FilterPages(object):
    def __init__(self, df):
        self.filtered_df = self.filter_url(df)
        
    def filter_url(self, df):
        """
        pages_to_keep_df = df \
            .groupby(df.pagelocation).agg(func.count("*").alias("count_pages"))\
                .where(pages_to_keep_df.count_pages > 5000).select("pagelocation")
                
        df = df.join(pages_to_keep_df \
                , pages_to_keep_df.pagelocation == df.pagelocation \
                , "left_semi")
        return df
        """
        temp_df = df.groupby(df.pagelocation).agg(func.count("*").alias("count_pages"))
        temp_df = temp_df.where(temp_df.count_pages > 1500).select("pagelocation")
        # usually 3000
                
        df = df.join(temp_df \
                , temp_df.pagelocation == df.pagelocation, "left_semi")  
        return df
    
    def run(self):
        return self.filtered_df