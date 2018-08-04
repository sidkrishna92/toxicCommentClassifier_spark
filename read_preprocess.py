class read_preprocess(object):
	
	def __init__(self, filename, spark):
		from pyspark.sql.functions import *
		self.filename = filename
		self.spark = spark
		# self.data_df = pd.DataFrame()
		self.read_file()
		self.preprocess()
		#self.stopwords()
	
	def read_file(self):
		import pandas as pd
		dfp = pd.read_csv(self.filename)
		dfp = dfp.replace({r'\n': ' '}, regex=True)
		dfp['comment_text'] = dfp['comment_text'].str.lower()

		# Comment this out if stop words works as UDF in spark-submit
		stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',\
					  'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',\
					  'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',\
					  'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',\
					  'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',\
					  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',\
					  'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',\
					  'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',\
					  'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',\
					  'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',\
					  'further', 'was', 'here', 'than']
		dfp['comment_text'] = dfp['comment_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
		self.data_df = self.spark.createDataFrame(dfp)
		del dfp
		
	def preprocess(self):
		from pyspark.sql.functions import *
		# self.data_df = self.data_df.withColumn('comment_text', lower(col('comment_text')))
		self.data_df = self.data_df.withColumn('comment_text', trim(self.data_df.comment_text))
		self.data_df = self.data_df.withColumn('comment_text', regexp_replace(self.data_df.comment_text, "\p{Punct}", " "))
	
	def corpus_featureGenerator(self, numFeatures, inp_colName, op_colName):
		from pyspark.ml import Tokenizer, HashingTF, IDF
		
		#Tokenize input text to be analyzed
		# tf(t,d) = #(t) in document 'd'
		tokenizer = Tokenizer(inputCol=inp_colName, outputCol="token")
		self.data_df = tokenizer.transform(self.data_df)
		
		#Generate Term-frequency vector
		# tf(t,d) = #(t) in document 'd'
		# df(t,D) = # docs that contain term 't'
		# D -- Corpus, d-- Document, t-- term
		termFreq = HashingTF(inputCol="token", outputCol="rawFeatures", numFeatures= numFeatures)
		self.data_df = termFreq.transform(self.data_df)
		
		#Generate Inverse-doc Freq vector
		# idf(t,D) = log( (|D|+1)/(df(t,D) +1) )
		idf = IDF(inputCol="rawFeatures", outputCol = op_colName)
		idfModel = idf.fit(self.data_df)
		self.data_df = idfModel.transform(self.data_df)

		
	# ## Spark Function to filter stop words. However, throws error while using
	# # with 'spark-submit'. For now, used pandas to filter Stop words.
	# # ERROR:: pickle.PicklingError: Could not serialize object: Py4JError: An error 
	# # occurred while calling o53.__getnewargs__. 
	# def stopwords(self):
		# remove_stops_udf = udf(self.remove_stops)
		# self.data_df = self.data_df.withColumn('comment_text', self.remove_stops_udf(self.data_df['comment_text']))
		
	# ## Use this Python Function to pass as a UDF to pyspark DF	
	# def remove_stops(str):
		# stop_words = []
		# pos = 0
		# cleaned_str = ''
		# text = str.split()
		# for word in text:
			# if word not in stop_words:
				# if pos==0:
					# cleaned_str = word
				# else:
					# cleaned_str += ' ' + word
				# pos += 1
		# return cleaned_str