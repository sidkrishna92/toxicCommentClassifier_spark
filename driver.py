'''
TOXIC COMMENT CLASSIFICATION CHALLENGE
Created by: Siddharth Krishnamurthy  
Date: 2018.07.27

Description:
This program builds a model to classify toxic comments based on a training dataset
from Wiki's talk page edits. This code was developeed for the Kaggle Toxic Comment 
Classification project hosted by Conversation AI

Input:

Output:


'''

def driver():
	from pyspark import SparkContext, SparkConf
	from pyspark.sql import SQLContext, HiveContext, SparkSession				
	spark = SparkSession.builder.master("yarn").appName('Comment_Classification').enableHiveSupport().getOrCreate()
	spark.sparkContext.setLogLevel('ERROR')
	filename = './spitzer/train.csv'
	
	import read_preprocess as rp
	train_data = rp.read_preprocess(filename, spark)
	# train_data.stopwords():
	
	print("\n"+ "--"*5)
	train_data.data_df.show()
	print("\n"+ "--"*5)
	train_data.data_df.printSchema()
	print("\n"+ "--"*5)
	print("Total number of rows: ", train_data.data_df.count())
	

	
	
if __name__ == "__main__":
	driver()

