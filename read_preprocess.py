class read_preprocess(object):
	
	
	
	def __init__(self, filename, spark):
		self.filename = filename
		self.spark = spark
		# self.data_df = pd.DataFrame()
		self.read_file()
		self.preprocess()
		
	def read_file(self):
		import pandas as pd
		dfp = pd.read_csv(self.filename)
		dfp = dfp.replace({r'\n': ' '}, regex=True)
		self.data_df = self.spark.createDataFrame(dfp)
		del dfp
		
	
	def preprocess(self):
		from pyspark.sql.functions import *
		import string
		self.data_df = self.data_df.withColumn('comment_text', lower(col('comment_text')))
		self.data_df = self.data_df.withColumn('comment_text', trim(self.data_df.comment_text))
		self.data_df = self.data_df.withColumn('comment_text', regexp_replace(self.data_df.comment_text, "\p{Punct}", " "))