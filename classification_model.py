class classification_model(object):
	def __init__(self, data_df, spark, feat_vec, target, model_name):
		self.df = data_df
		self.spark = spark
		self.feat_vec = feat_vec
		self.target = target
		self.model_name = model_name
		self.data_split()
		self.train_model()
			
	def data_split(self):
		(self.trainingDf, self.testDf) = self.df.randomSplit([0.7, 0.3],\
		seed = 100)
		print("\n" + "--"*5)
		print("Training Dataset Count: " + str(trainingDf.count()))
		print("Validation Dataset Count: " + str(testDf.count()))
	
	def train_model(self):
		
		if self.model_name == 'logistic':
			print("\n" + "--"*5)
			print("Fitting LR model for %s"%self.target)
			from pyspark.ml.classification import LogisticRegression
			lr = LogisticRegression(maxIter=30, regParam=0.3, elasticNetParam=0, featuresCol='features', labelCol = self.target)
			self.model = lr.fit(self.trainingData)	
			
			
	def eval_model(self):
		from pyspark.ml.evaluation import BinaryClassificationEvaluator
		self.predictions = self.model.transform(self.testData)
		evaluator = BinaryClassificationEvaluator(labelCol=self.target)
		self.auc = evaluator.evaluate(self.predictions, {evaluator.metricName: "areaUnderROC"})
		print("\nArea Under ROC for Validation set: %f"% self.auc)
		