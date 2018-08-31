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
	
	#Input Vars
	filename_train = './spitzer/train.csv'
	filename_test = './spitzer/test.csv'
	model_name = 'logistic'
	target = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
	feat_vec = 'features'
	numFeatures=10000
	inp_colName="comment_text"
	op_colName= "features"
	
	#Import classes
	import read_preprocess as rp
	import classification_model as cm
	import model_tester as tm
			
	train_data = rp.read_preprocess(filename_train, spark)
	train_data.corpus_featureGenerator(numFeatures, inp_colName, op_colName)
	
	test_data = rp.read_preprocess(filename_test, spark)
	test_data.corpus_featureGenerator(numFeatures, inp_colName, op_colName)
	# print("\n"+ "--"*5)
	# train_data.data_df.show()
	# print("\n"+ "--"*5)
	# train_data.data_df.printSchema()
	print("\n"+ "--"*5)
	print("Train Set number of rows: ", train_data.data_df.count())
	print("\n"+ "--"*5)
	print("Test Set number of rows: ", test_data.data_df.count())
	
	for i,label in enumerate(target):
		#Build and evaluate model on training/validation set
		model_build = cm.classification_model(train_data.data_df, spark, feat_vec, label, model_name)
		model_build.eval_model()

		#Run model on Test Data
		test_model = tm.model_tester(test_data.data_df, spark, feat_vec, label, model_name, model_build.model) 
		final_df = test_model.pred_df.withColumn(label, test_model.pred_df["probability"])
	
	#final_df.coalesce(1).write.csv("./spitzer/results/")
	
if __name__ == "__main__":
	driver()

