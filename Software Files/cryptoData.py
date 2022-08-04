import plotly.graph_objects as go #allows us to plot data graphically
import matplotlib.pyplot as plt
import json #allows us to convert json object to a python object
import pandas as pd #stands for Python Data Analysis library
from datetime import datetime #allows for date/time operations
import csv #allows us to read/modify csv file
import numpy as np





def dogeStuffs():
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
    for i in range(1, 28):
        if(i<10):
            dateString = '02/0'
        else:
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
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom')


    #closing price chart
    plot1 = plt.figure(2)
    #setting labels
    plt.title('DOGE Daily Closing Price')
    plt.xlabel('Dates')
    plt.ylabel('Price (in cents)')
    #plt.plot plots a line
    priceBars = plt.bar(datesX, dogecoin_dataframe['close'])

    #displays values on top of bar
    for rect in priceBars: 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.4f' % float(height), ha='center', va='bottom')

    #shows both bar charts
    plt.show()
    
    

def bitcoinStuffs():
    #our arrays we will use to store data. Will later be used to store into panda dataframe
    dates = []
    high_data = []
    low_data = []
    open_data = []
    close_data = []
    volume_data = []
    market_cap = []

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
    for i in range(1, 28):
        if(i<10):
            dateString = '02/0'
        else:
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

    #closing price chart
    plot1 = plt.figure(2)
    #setting labels
    plt.title('BTC Daily Closing Price')
    plt.xlabel('Dates')
    plt.ylabel('Price (in dollars)')

    
    #plt.plot plots a line
    priceBars = plt.bar(datesX, bitcoin_dataframe['close'])

    #displays values on top of bar
    for rect in priceBars: 
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f' % float(height), ha='center', va='bottom', fontsize=7.5)

    #shows both bar charts
    plt.show()
    
    


dogeStuffs()
bitcoinStuffs()