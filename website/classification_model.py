# Importing libraries
import pandas as pd
import nltk
import demoji
demoji.download_codes()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from collecting_tweets import MLProcessor

#reading dataset from csv file
cols = [ 'Tweet','Polarity']
dataset = pd.read_csv("website/static/Dataset/Dataset40K.csv", encoding='latin_1', skipinitialspace=True, usecols=cols)
print ('dataset reading finished')

mlProcessor = MLProcessor()

# Search for tweets
twitterDataFrame = mlProcessor.searchTweets("call of duty")
print("searchTweets is Done")
pd.options.display.max_colwidth = 1000
print(twitterDataFrame["Tweet"])

# Twitter Pre-Process
mlProcessor.TwitterPreProcess(twitterDataFrame)
print ('Twitter Pre-Process is finished')

# English Pre-Process
mlProcessor.EnglishPreProcess(twitterDataFrame)
print ('English Pre-Process is finished')

#converting from dataframe to list of tweets
searchTweetsList = mlProcessor.TextPreperationPreProcess(twitterDataFrame)
print ('English Pre-Process for searchTweetsList is finished')

#converting from dataframe to list of tweets
datasetTweetsList = mlProcessor.TextPreperationPreProcess(dataset)
print ('English Pre-Process for datasetTweetsList is finished')

mergeArrays = searchTweetsList + datasetTweetsList

vectorizedTweets = mlProcessor.TextVectorizer(mergeArrays)

vecSearchTweetsList = vectorizedTweets[:len(searchTweetsList)]

vecDatasetTweetsList = vectorizedTweets[len(searchTweetsList):]

classifier = mlProcessor.SvmClassifier(vecDatasetTweetsList, dataset['Polarity'])

predictions = classifier.predict(vecSearchTweetsList)
print(twitterDataFrame["Tweet"])
print(predictions)