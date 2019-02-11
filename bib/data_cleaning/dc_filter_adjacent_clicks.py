from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag

class FilterAdjacentClicks(object):
    def __init__(self, df):
        self.filtered_df = self.filter_adjacent_clicks(df)
        
    def filter_adjacent_clicks(self,df):
        window = Window.partitionBy("sessionnumber").orderBy("eventtimestamp")
        duplicate_window = col("pagelocation") == lag("pagelocation", 1).over(window)
        df_out = df.withColumn("diff", duplicate_window).fillna({'diff':False})
        return df_out.where(df_out.diff == False).drop("diff")
    
    def run(self):
        return self.filtered_df
