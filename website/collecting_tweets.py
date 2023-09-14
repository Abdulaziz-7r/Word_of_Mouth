# Importing libraries
import tweepy
import pandas as pd
import re
import nltk
import demoji
demoji.download_codes()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from better_profanity import profanity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics


class MLProcessor:
  def __init__(self):
    # Setting up Twitter API 
    self.api_key = ""
    self.api_key_sec = ""
    self.access_token = ""
    self.access_token_sec = ""
    self.auth_handler = tweepy.OAuthHandler(consumer_key=self.api_key, consumer_secret= self.api_key_sec)
    self.auth_handler.set_access_token(self.access_token, self.access_token_sec)
    self.api = tweepy.API(self.auth_handler)

  # Twitter Pre-Process
  @staticmethod
  def TwitterPreProcess(data):
    for index,row in data.iterrows():

      sentence = row['Tweet']
      sentence = re.sub(r"http\S+", '', sentence) # removing URLs
      sentence = re.sub("@[A-Za-z0-9_]+","", sentence) # removing mentions 
      sentence = re.sub("#[A-Za-z0-9_]+","", sentence) # removing hashtags 
      sentence = re.sub(r'\d+',"", sentence) #removing digits
      sentence = profanity.censor(sentence) # censoring bad languages
      sentence = re.sub('[@|#$^&+\-%*/=!>()_{}~<>?.,:;\"½]',"", sentence) #removing special characters  
      # replacing emoji with it meaning
      emoji = demoji.findall(sentence)
      key_list = list(emoji.keys())
      val_list = list(emoji.values())

      j = 0
      while j < len(key_list):
        sentence = re.sub(key_list[j]," " + val_list[j]+ " ", sentence)
        j += 1  

      data.loc[index, 'Tweet'] = sentence # replacing the tweet after filtering
  
  # English Pre-Process
  @staticmethod
  def EnglishPreProcess(data):
    # English Pre-Process
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    data['Tweet'] = data['Tweet'].str.lower() # changing tweets to lowercase

    for index,row in data.iterrows():

      filter_sentence = []
      sentence = row['Tweet']
      sentence = re.sub(r'[^\w\s]',' ', sentence) ###########
      sentence = re.sub(r'(.)\1\1+', r'\1\1', sentence) # Replace 3 or more consecutive letters by 2 letter. ##################################################
      words = nltk.word_tokenize(sentence)
      words = [w for w in words if not w in stop_words]
      
      for word in words:
        if len(word) <= 1:
          continue
        filter_sentence.append(lemmatizer.lemmatize(word))
      filter_sentence = (" ").join(filter_sentence)
      data.loc[index, 'Tweet'] = filter_sentence
  
  # converting from dataframe to list of tweets
  @staticmethod
  def TextPreperationPreProcess(data):
    tokenizedTweets = data['Tweet'].values.tolist()
    return tokenizedTweets

  # Collecting Tweets from twitter based on search word
  def searchTweets(self, searchText):
    query = "\"" +searchText+ "\"" + '-filter:retweets'
    tweet_amount = 20
    data = []
    tweets = tweepy.Cursor(self.api.search_tweets, q = query, lang='en').items(tweet_amount)

    for tweet in tweets:
      data.append([tweet.text, ""])

    columns = ['Tweet', 'Polarity']
    twitterDataFrame = pd.DataFrame(data, columns=columns)
    return twitterDataFrame

  @staticmethod
  def TextVectorizer(tweetsList):
    vectorizer = CountVectorizer()
    vecTweets = vectorizer.fit_transform(tweetsList)
    tweetsToPredict = vecTweets.toarray()
    return tweetsToPredict
  
  @staticmethod
  def KnnClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=8)
    classifier = KNeighborsClassifier(n_neighbors=13)
    classifier.fit(X_train, y_train)

    #result of 1st classifier
    predictions = classifier.predict(X_test)

    print("result of 1st classifier")
    print(classifier.score(X_test, y_test))

    print (metrics.confusion_matrix(y_test,predictions))
    return classifier

  @staticmethod
  def SvmClassifier(X, y):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=8)
      classifier = svm.LinearSVC(C = .05)
      classifier.fit(X_train, y_train)

      #result of SVM classifier
      predictions = classifier.predict(X_test)

      print("result of SVM classifier")
      print(classifier.score(X_test, y_test))

      print (metrics.confusion_matrix(y_test,predictions))
      return classifier
      