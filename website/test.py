import unittest
from collecting_tweets import MLProcessor



class Test(unittest.TestCase):

    def test_serachTweets(self):
        mlProcessor = MLProcessor()
        tweetsDataFrame = mlProcessor.searchTweets('pepsi')
        self.assertEqual(tweetsDataFrame.size, 40)

    def test_TextVectorizer(self):
        mlProcessor = MLProcessor()
        tweetsList = ['really feel like getting today got study tomorrow practical exam', 'oh sorry think retweeting','leaving parking lot work']
        vectorizedTweets = mlProcessor.TextVectorizer(tweetsList)
        length = len(vectorizedTweets[0])
        self.assertEqual(length, 18)



if __name__ == '__main__':
    unittest.main()