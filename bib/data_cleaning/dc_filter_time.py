import pyspark.sql.functions as func

class FilterTime(object):
    """
    Normalizes time
    @param spark        : spark session
    @param df           : dataframe holding the target value, output from data selection step
    """
    def __init__(self, spark, df \
    , database_name="34np_project_celebrus_ku", pagesummary_name="hd_wt_pagesummary"):
        self.spark = spark
        self.df = df
        self.pagesummary_table = spark.table(database_name + "." + pagesummary_name)

    def run(self):
        df_joined_pageviewtimeactive = self.add_pageviewactivetime()
        #print("after join")
        #print(df_joined_pageviewtimeactive)
        normalized_time = self.normalize_time(df_joined_pageviewtimeactive)
        #print("normtime")
        #print(normalized_time)
        return normalized_time

    def normalize_time(self, df):
        accumulated_time = df \
            .groupBy(df.target) \
            .agg(func.sum(df.pageviewactivetime).alias("sum_pagetimeviewactivetime"))

        df_w_sum_time = df.join(accumulated_time \
                , df.target == accumulated_time.target
                , 'inner') \
            .select(df["*"], accumulated_time.sum_pagetimeviewactivetime)

        df_normalized_time = df_w_sum_time \
            .withColumn("normalized_pagetimeviewactivetime" \
                , df_w_sum_time.pageviewactivetime / df_w_sum_time.sum_pagetimeviewactivetime)

        df_normalized_time = df_normalized_time \
            .drop(*["sum_pagetimeviewactivetime" \
                , "pageinstanceid"])

        print("Count of normalized_time: ", df_normalized_time)
        return df_normalized_time



    def add_pageviewactivetime(self):
        df_joined = self.df.join(self.pagesummary_table \
                , self.df.pageinstanceid == self.pagesummary_table.pageinstanceid \
                , 'inner') \
            .select(self.df["*"], self.pagesummary_table.pageviewactivetime)
        #print("count of addpageview: ", df_joined.count())

        return df_joined
