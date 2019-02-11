from pyspark.sql.functions import col, regexp_replace, lower

class CleanPageLocation(object):
    def __init__(self, df):
        self.filtered_df = self.filter_url(df)
        
    def filter_url(self,df):
        """
        \s+                : remove whitespacecharacters, possibly left over
        \?(.*)             : match character ? and everything afterwards
        [#|&]              : match last character #, &
        ^(https|http|www.) :  
        \/\/               :
        
        .withColumn('pagelocation', regexp_replace('pagelocation','https:\/\/',''))\
                 .withColumn('pagelocation', regexp_replace('pagelocation','http:\/\/',''))\
                 .withColumn('pagelocation', regexp_replace('pagelocation','^www.',''))\
                 
        
        """
        return df.withColumn('pagelocation', regexp_replace('pagelocation', '\s+', '/'))\
                 .withColumn('pagelocation', regexp_replace('pagelocation', '\?(.*)', ''))\
                 .withColumn('pagelocation', regexp_replace('pagelocation','[#|&]$',''))\
                 .withColumn('pagelocation', regexp_replace('pagelocation', '^(https|http|www.)',''))\
                 .withColumn('pagelocation', regexp_replace('pagelocation','\/\/','/'))\
                 .withColumn('pagelocation', regexp_replace('pagelocation',"[\/|,|.|'|-]$",''))\
                 .withColumn('pagelocation',lower(col("pagelocation")))\
                 .where(col("pagelocation").like("%.dk%"))

    def get_filtered(self):
        return self.filtered_df
    
    
#.withColumn('pagelocation', regexp_replace('pagelocation','.aspx$',''))\