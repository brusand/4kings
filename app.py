import os, csv
import talib
import pandas
from flask import Flask, escape, request, render_template
from configobj import ConfigObj

app = Flask(__name__)


@app.route('/')
def index():

    pattern  = request.args.get('pattern', False)
    config = ConfigObj('./config/4kings.ini')
    tickers = ConfigObj('./config/tickers-' + config.get('user') + '.ini')


    if pattern:
        for filename in os.listdir('backtest/1d'):
            df = pandas.read_csv('backtest/1d/{}'.format(filename))
            pattern_function = getattr(talib, pattern)
            ticker = filename.split('_')[0]

            try:
                results = pattern_function(df['open'], df['high'], df['low'], df['close'])
                last = results.tail(1).values[0]

                if last > 0:
                    tickers[ticker][pattern] = 'bullish'
                elif last < 0:
                    tickers[ticker][pattern] = 'bearish'
                else:
                    tickers[ticker][pattern] = None

                tickers.write()

            except Exception as e:
                print('failed on filename: ', filename)

    return render_template('index.html',  tickers=tickers.keys(), timeframes=['1d','1w'], zones=['MISSEDTRAIN','FISRCLASS', 'ECOCLASS', 'DYING'])
