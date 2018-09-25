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
	from pyspark.sql.functions import monotonically_increasing_id, lit
	from pyspark.sql.types import FloatType

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
	
	def ith_(v, i):
                try:
                        return float(v[i])
                except ValueError:
                        return None

	ith = spark.udf.register("ith", ith_, FloatType())
	final_df = test_data.data_df.select('comment_text')
	final_df = final_df.withColumn('row_index', monotonically_increasing_id())
	
	for i,label in enumerate(target):
		#Build and evaluate model on training/validation set
		model_build = cm.classification_model(train_data.data_df, spark, feat_vec, label, model_name)
		model_build.eval_model()

		#Run model on Test Data
		test_model = tm.model_tester(test_data.data_df, spark, feat_vec, label, model_name, model_build.model) 
		test_model.pred_df = test_model.pred_df.withColumn('row_index', monotonically_increasing_id())

		#test_model.pred_df.show()
		test_model.pred_df = test_model.pred_df.withColumn("p1", ith("probability", lit(0)))
		test_model.pred_df = test_model.pred_df.drop("probability")

		final_df = final_df.join(test_model.pred_df, on=["row_index"]).sort("row_index").withColumnRenamed("p1", label)

	final_df = final_df.drop("row_index", "comment_text")
	final_df.show()
	final_df.coalesce(1).write.csv("./spitzer/results/")
	
if __name__ == "__main__":
	driver()

