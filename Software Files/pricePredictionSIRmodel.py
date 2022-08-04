import json
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import datetime as dt
import requests
import emoji
import spacy
import plotly.graph_objects as go #allows us to plot data graphically
import matplotlib.pyplot as plt
import numpy as np

#json file with doge comments
doge = '/Users/sem/VScode python/crypto price model/dogeComments.json'

#json file with btc comments
btc = '/Users/sem/VScode python/crypto price model/bitcoinComments.json'


#02/13 closing price
bitcoinStartingPrice = 47105.52
dogeStartingPrice = 0.0664

startingPrice = bitcoinStartingPrice
#startingPrice = dogeStartingPrice

#scale factor for positive sentiment
scaleFactors = []
#sentiment scores
scores = []
#predicted price
predicted = []


#contains bitcoin comments from February 12 to February 26th
def cryptoSentiments(cryptoComments, date):

    #convert to string object
    cryptoComments = [str(i) for i in cryptoComments]
    cryptoCommentsUncleaned = ',  '.join(cryptoComments)
    cryptoCommentsUncleaned

    #remove emojis
    cryptoEmojiDelete = emoji.get_emoji_regexp().sub(u'', cryptoCommentsUncleaned)

    #tokenizer breaks apart every word in the string into individual words (tokens)
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
    tokenized_string = tokenizer.tokenize(cryptoEmojiDelete)

    #converts tokens to all lowercase
    lower_string_tokenized = [word.lower() for word in tokenized_string]

    #removing stopwords
    nlp = spacy.load('en_core_web_sm')
    all_stopwords = nlp.Defaults.stop_words
    text = lower_string_tokenized
    tokens_without_sw = [word for word in text if not word in all_stopwords]

    #lemmatizing: trim words down to their root words. stemming might not return a real word
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = ([lemmatizer.lemmatize(w) for w in tokens_without_sw])

    #calculating sentiment using VADER 
    cleaned_output = lemmatized_tokens

    sia = SIA()
    results = []
    for sentences in cleaned_output:
        pol_score = sia.polarity_scores(sentences)
        pol_score['words']=sentences
        results.append(pol_score)

    pd.set_option('display.max_columns', None, 'max_colwidth', None)
    df = pd.DataFrame.from_records(results)

    df['label'] = 0
    df.loc[df['compound'] >= 0.05, 'label'] = 1
    df.loc[df['compound'] <= -0.05, 'label'] = -1
    df.head()


    df_positive = df.loc[df['label']==1]
    df_negative = df.loc[df['label']==-1]
    df_neutral = df.loc[df['label']==0]


    #dateframe of counts of all labels
    #counts = df.label.value_counts()

    #dateframe of counts of all positive labels
    counts = df_positive.label.value_counts()
    
    #gets scale factor for each positive sentiment score according to total score
    scaleFactor = counts/len(results)

    #appends to list of scaleFactors; used to find scaled positive sentiment scores
    scaleFactors.append(scaleFactor)

    return len(results)

def predictPrice(file, startingPrice):
    with open(file) as cryptoFile:
        data = json.load(cryptoFile)

    time = 1613106000 #02/12

    averageScore = 0 

    #grabs 100 comments from each day and sends into sentiment analyzer function that will return scores
    for i in range(15):
        cryptoComments = []
        for data1 in data['data']:
            if (data1['created_utc'] < time + 86400) and (data1['created_utc'] >= time) :
                body = data1['body']
                date = dt.datetime.fromtimestamp(data1['created_utc'])
                cryptoComments.append(body)
        time +=86400
        averageScore+=cryptoSentiments(cryptoComments, date)
    
    #average of total sentiment scores from all dates
    #for these 2 weeks in Feb: 2586 average total scores
    averageScore = averageScore/15

    predictedPrice = 0

    #calculates price prediction
    for i in range(15):
        if i < 14:
            if file==doge:
                #taking into account for low doge value, scale the change in price by 10
                predictedPrice = startingPrice + (10*(scaleFactors[i+1] - scaleFactors[i])*startingPrice)
            if file==btc:
                predictedPrice = startingPrice + ((scaleFactors[i+1] - scaleFactors[i])*startingPrice)  
            predicted.append(predictedPrice)
            startingPrice = predictedPrice

    #stores list of scaled positive sentiment scores into list
    for i in range(len(scaleFactors)):
        scaleFactors[i] = scaleFactors[i]* averageScore
    
    #used to allow last sentiment score to be 0 for 02/27
    temp = scaleFactors[1]*0

    for i in range(2,15):
        scores.append(scaleFactors[i])
    scores.append(temp)

def dogeVisualization(scores, predicted):
    #our arrays we will use to store data. Will later be used to store into panda dataframe
    dates = []
    high_data = []
    low_data = []
    open_data = []
    close_data = []
    volume_data = []
    market_cap = []

    #opens our historical data csv file in 'read' mode
    with open('/Users/sem/VScode python/crypto price model/dogecoinHistoricalData.csv', 'r') as dogeFile: 
        #csv.DictReader allows us to access our data by keys: 'high', 'low', etc
        dogeData = csv.DictReader(dogeFile)

        #steps over to the next line/row
        #this excludes the column name when we input our data
        #next(dogeData)

        #initializing data into our arrays 
        for row in dogeData:
            dates.append(row['Date'])
            high_data.append(float(row['High']))
            low_data.append(float(row['Low']))
            open_data.append(float(row['Open']))
            close_data.append(float(row['Close']))
            volume_data.append(int(row['Volume']))
            market_cap.append(int(row['Marketcap']))

    #creates panda data frame with our define dcolumns
    dogecoin_dataframe = pd.DataFrame(columns = ['dates', 'high', 'low', 'open', 'close', 'volume', 'market_cap'])

    #initializes our data frame columns with values from our data arrays
    dogecoin_dataframe['dates'] = dates
    dogecoin_dataframe['high'] = high_data
    dogecoin_dataframe['low'] = low_data
    dogecoin_dataframe['open'] = open_data
    dogecoin_dataframe['close'] = close_data
    dogecoin_dataframe['volume'] = volume_data
    dogecoin_dataframe['market_cap'] = market_cap

    #converts date into smaller form factor for graph representation
    datesX = []
    dateString = '02/'
    for i in range(14, 28):
        dateString ='02/'
        datesX.append(dateString + str(i))

    #scaling down volume numbers to show billions; i.e 6.4B
    volumeY = []
    for i in range(len(volume_data)):
        volumeY.append(volume_data[i]/1000000000)

    #volume chart
    plot1 = plt.figure(1)
    #setting labels
    plt.title('DOGE Daily Volume')
    plt.xlabel('Dates')
    plt.ylabel('Volume (in billions)')

    #plt.plot plots a line
    volumeBars = plt.bar(datesX, volumeY)

    #displays values on top of bar
    for rect in volumeBars: 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.4f' % float(height), ha='center', va='bottom')


    ind = np.arange(len(datesX))
    df = pd.DataFrame(np.c_[ predicted, close_data], index = datesX, columns = [ 'Predicted', 'Actual'])

    ax = df.plot.bar(y=['Predicted', 'Actual'], title='Doge Daily Closing Price')
    
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(),decimals=4),(p.get_x()+p.get_width()/2., p.get_height()),ha='center',va='center',  xytext=(0, 10),  textcoords='offset points')

    plt.show()

def bitcoinVisualization(scores, predicted):
    #our arrays we will use to store data. Will later be used to store into panda dataframe
    dates = []
    high_data = []
    low_data = []
    open_data = []
    close_data = []
    volume_data = []
    market_cap = []

#'/Users/sem/VScode python/crypto price model/bitcoinHistoricalData.csv'

    #opens our historical data csv file in 'read' mode
    with open('/Users/sem/VScode python/crypto price model/bitcoinHistoricalData.csv', 'r') as btcFile: 
        #csv.DictReader allows us to access our data by keys: 'high', 'low', etc
        btcData = csv.DictReader(btcFile)

        #steps over to the next line/row
        #this excludes the column name when we input our data
        #next(dogeData)

        #initializing data into our arrays 
        for row in btcData:
            dates.append(row['Date'])
            high_data.append(float(row['High']))
            low_data.append(float(row['Low']))
            open_data.append(float(row['Open']))
            close_data.append(float(row['Close']))
            volume_data.append(float(row['Volume']))
            market_cap.append(float(row['Marketcap']))

    #creates panda data frame with our define dcolumns
    bitcoin_dataframe = pd.DataFrame(columns = ['dates', 'high', 'low', 'open', 'close', 'volume', 'market_cap'])

    #initializes our data frame columns with values from our data arrays
    bitcoin_dataframe['dates'] = dates
    bitcoin_dataframe['high'] = high_data
    bitcoin_dataframe['low'] = low_data
    bitcoin_dataframe['open'] = open_data
    bitcoin_dataframe['close'] = close_data
    bitcoin_dataframe['volume'] = volume_data
    bitcoin_dataframe['market_cap'] = market_cap
    

    #converts date into smaller form factor for graph representation
    datesX = []
    dateString = '02/'
    for i in range(14, 28):
        dateString ='02/'
        datesX.append(dateString + str(i))

    #scaling down volume numbers to show billions; i.e 6.4B
    volumeY = []
    for i in range(len(volume_data)):
        volumeY.append(volume_data[i]/1000000000)

    #volume chart
    plot1 = plt.figure(1)
    #setting labels
    plt.title('BTC Daily Volume')
    plt.xlabel('Dates')
    plt.ylabel('Volume (in billions)')

    #plt.plot plots a line
    volumeBars = plt.bar(datesX, volumeY)

    #displays values on top of bar
    for rect in volumeBars: 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom')


    ind = np.arange(len(datesX))
    df = pd.DataFrame(np.c_[scores, predicted, close_data], index = datesX, columns = ['scores', 'Predicted', 'Actual'])

    ax = df.plot.bar(y=['scores','Predicted', 'Actual'], title='BTC Daily Closing Price')
    
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(),decimals=2),(p.get_x()+p.get_width()/2, p.get_height()),ha='center',va='center',  xytext=(0, 10),  textcoords='offset points')

    plt.show()
    

predictPrice(btc, startingPrice)   
bitcoinVisualization(scores, predicted)

#predictPrice(doge, startingPrice)   
#dogeVisualization(scores, predicted)


