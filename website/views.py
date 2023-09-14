from flask import Blueprint, render_template, request, flash, jsonify, url_for, redirect
from flask_login import login_required, current_user
from .models import Product, Sentiment
from . import db
import json 
from .collecting_tweets import MLProcessor
import pandas as pd

views = Blueprint('views', __name__)

@views.route('/', methods= ['GET', 'POST'])
@login_required
def main_menu():

    return render_template("main_menu.html", user=current_user)
 
@views.route('/search', methods= ['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        product = request.form.get('product')
        duration = request.form.get('duration')
        if len(product) < 1:
            flash('Please enter the product name', category='error')
        else:
            new_product = Product(name = product, user_id = current_user.id)
            db.session.add(new_product)
            db.session.commit()
            flash('Product added!', category='success')

            #reading dataset from csv file
            cols = [ 'Tweet','Polarity']
            dataset = pd.read_csv("website/static/Dataset/Dataset40K.csv", encoding='latin_1', skipinitialspace=True, usecols=cols)
            print ('dataset reading finished')

            # performing query to select the product based on the name    
            productObj = Product.query.filter_by(name=product).first()
            

            mlProcessor = MLProcessor()
            twitterDataFrame = mlProcessor.searchTweets(productObj.name)
            print("Product Name: "+productObj.name)

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
            print('Tweets: '+twitterDataFrame["Tweet"])
            print(predictions)

            #Adding tweets to database#
            for index,row in twitterDataFrame.iterrows():
                sentence = row["Tweet"]
                polarity = str(predictions[index])
                new_sentiment = Sentiment(tweet=sentence, polarity=polarity, product_id=productObj.id)
                db.session.add(new_sentiment)
        
            db.session.commit()    
            flash('All the Sentiments are added!', category='success')

            return redirect(url_for("views.main_menu"))

    return render_template("search.html", user=current_user)

@views.route('/history', methods=['GET', 'POST'])
@login_required
def history():
    if request.method == 'POST':
        productId = request.form.get('productId')
        return redirect(url_for('views.chart', productId = productId))

    return render_template("history.html", user=current_user)

@views.route('/chart/<productId>')
@login_required
def chart(productId):
    sentimentObj = Sentiment.query.filter_by(product_id = productId)
    positive = 0
    negative = 0
    tweetsList = []
    
    for sentiment in sentimentObj:
        if sentiment.polarity == '0':
            negative += 1
            tweetsList.append([sentiment.tweet, "negative"])
        else:
            positive += 1    
            tweetsList.append([sentiment.tweet, "positive"])
    columns = ['Tweet', 'Polarity']
    TweetDataFrame = pd.DataFrame(tweetsList, columns=columns)
    return render_template("chart.html", user=current_user, positive=positive, negative=negative, TweetDataFrame = TweetDataFrame)        
