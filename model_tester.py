class model_tester(object):
	
	def __init__(self, data_df, spark, feat_vec, target, model_name, classification_model):
		self.df = data_df
		self.spark = spark
		self.feat_vec = feat_vec
		self.target = target
		self.model_name = model_name
		self.model = classification_model
		self.test_model()
	
	def test_model(self):
		
		predictions = self.model.transform(self.df)
		self.pred_df = predictions.select('probability')