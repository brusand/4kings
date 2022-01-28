from __future__ import unicode_literals
import asyncio
import aiohttp
import argparse
from configobj import ConfigObj
from binance.client import Client
from datetime import datetime as dtime, date, time, timedelta
from prompt_toolkit import __version__ as ptk_version
PTK3 = ptk_version.startswith('3.')
import threading
from prompt_toolkit import PromptSession
from prompt_toolkit import prompt

from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.history import FileHistory
import ccxt
import datetime
import os
import os.path
import pandas as pd
from time import sleep
from binance import AsyncClient, Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from concurrent.futures import ThreadPoolExecutor

from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from Telegram import Telegram;
import os
import numpy as np
import importlib
from models.price import Price
from models.order import Order
from models.exchange import Exchange
from numpy.polynomial import Chebyshev as T
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
import arrow
from pandas import DataFrame

#         trade ROSE/BNB --timerange 1m -timestart 2022-01-12
ECOCLASS_START = 5 #0.618
ECOCLASS_END = 6 #0.786
FIRSTCLASS_START = 6 #0.786
FIRSTCLASS_END = 7 #0.886
Lmax = 0
L0114 = 1
L0214 = 2
L0382 = 3
L05 = 4
L0618 = 5
L0786 = 6
L0886 = 7
Lmin = 8
get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

class Trading():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='4kings')
        self.parser.add_argument('--exchange', nargs='?', help='Exchange', default='binance')
        self.parser.add_argument('--user', nargs='?', help='User', required=False)
        self.parser.add_argument('--key', help='key', required=False)
        self.parser.add_argument('--secret', nargs='?', help='secret', required=False)
        self.parser.add_argument('--fiat', nargs='?', help='fiat', required=False)
        self.parser.add_argument('--stake', nargs='?', help='stake', required=False)
        self.parser.add_argument('--cmde', nargs='?', help='Cmde', default='')
        self.parser.add_argument('--shell ', action="store_true", default=False)
        self.subparsers = self.parser.add_subparsers(dest='cmdla', help='sub-command help')



        self.parser_backtest = self.subparsers.add_parser('backtest')
        self.parser_backtest.add_argument('trade', nargs='?', action="store", default='ROSE/BNB')
        self.parser_backtest.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_backtest.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_backtest.add_argument('--dry', help='trend', required=False, default='FISRTCLASS')
        self.parser_backtest.add_argument('--timerange', help='time range', required=False, default='1m')
        self.parser_backtest.add_argument('--timestart', help='time start', required=False, default='21-01-2022_00:15')
        self.parser_backtest.add_argument('--timeto', help='time to', required=False)
        self.parser_backtest.add_argument('--debug', help='debug', action="store_true", default=False)
        self.parser_backtest.add_argument('--pausestart', help='time to', required=False, default='21-01-2022_00:35')
        self.parser_backtest.add_argument('--pausestop', help='time to', required=False, default='21-01-2022_01:50')
        self.parser_backtest.add_argument('--days', help='days', type=int, action="store", required=False, default=1)
        self.parser_backtest.add_argument('--hours', help='days', type=int, action="store", required=False, default=0)
        self.parser_backtest.add_argument('--minutes', help='days', type=int, action="store", required=False, default=0)
        self.parser_backtest.set_defaults(func=self.backtest)


        self.config = ConfigObj('./config/4kings.ini')
        if self.config.get('user') == None:
            self.config['users'] = {}

        else:
            self.tickers = ConfigObj('./config/tickers-' + self.config.get('user') + '.ini')
            self.tickers.write()

        self.config.write()
        self.telegram = Telegram( self.config )
        self.binance = ccxt.binance()

    def get_symbol(self):
        return self.currency + self.asset

    def init(self, args):
        print('init')
        if not args.user:
           self.user = ''
        else:
            if self.config['users'].get(args.user) == None:
                self.config['users'][args.user] = {}
                self.config['users'][args.user]['last_exchange'] = ''
                self.config['users'][args.user]['key'] = ''
                self.config['users'][args.user]['secret'] = ''
                self.config['users'][args.user]['user_name'] = ''
                self.config['users'][args.user]['exchange'] = 'binance'
                self.config['users'][args.user]['fiat'] = 'EUR'
                self.config['users'][args.user]['stake'] = 'USDT'
                self.config['users'][args.user]['dry'] = true

            if args.exchange is not  None:
                self.config['users'][args.user]['exchange'] = args.exchange

            if args.key is not  None:
                self.config['users'][args.user]['key'] = args.key

            if args.secret is not  None:
                self.config['users'][args.user]['secret'] = args.secret

            if args.user is not  None:
                self.config['users'][args.user]['user_name'] = args.user

            if args.fiat is not None:
                self.config['users'][args.user]['fiat'] = args.fiat

            if args.stake is not None:
                self.config['users'][args.user]['stake'] = args.stake

            self.user = args.user
            self.config['user'] = args.user
            self.config.write()

        self.key = self.config['users'][args.user]['key']
        self.secret = self.config['users'][args.user]['secret']
        self.user_name = self.config['users'][args.user]['user_name']
        self.fourkings = ConfigObj('./config/' + '4kings-'+ self.user + '.ini')

        self.bye = False

        exchangeId = self.config['users'][args.user]['main_exchange']
        exchangeClass = getattr(ccxt, exchangeId)
        self.exchange = exchangeClass({
            'apiKey': self.config['users'][args.user]['key'],
            'secret': self.config['users'][args.user]['secret'],
        })

        self.telegram._send_msg('*Welcome 4kings bot:*')
        self.init_process(args)

    def init_process(self, args):
        self.threads={};
        try:
            self.ps_restart(args)
        except Exception as _error:
            print('error',  _error)



    def get_histos(self, args):
        klines={}

        symbol = args.ticker+args.stake

        if symbol == '*':
            relevant = [symbol for symbol in self.get_local_tickers(args)]
        else:
            relevant = [symbol]

        print('get histo for ', relevant)

        untilThisDate = dtime.now()

        if self.tickers[symbol].get(args.timerange) != None:
            print(self.tickers[symbol][args.timerange])
            sinceThisDate = self.tickers[symbol][args.timerange]
        else:
            sinceThisDate = untilThisDate - timedelta(days=args.days)

        print('get_klines', symbol, sinceThisDate,  'START')
        print('date start : ', sinceThisDate, ' date to :', untilThisDate)


        dt = [sinceThisDate, untilThisDate]
        #callbacks = [lambda sym=symbol: self.ohlcv(dt, sym, args) for symbol in relevant]
        callbacks = [lambda sym=symbol: self.get_klines(sym, args) for symbol in relevant]

        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(callback) for callback in callbacks]
            for task in tasks:
                symbol, lines = task.result()
                klines[symbol] = lines

            returns, symbols = [], []
            for symbol in relevant:
                dataframe = pd.DataFrame(klines[symbol])
                if dataframe.size > 0:
                    print(symbol, args.timerange , ' to csv')
                    dataframe.columns = ['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore']

                    dataframe.to_csv('./data/' + args.timerange + '/' + symbol + '.csv',
                                 encoding='utf-8',
                                 index=False)

                    candle = dataframe.iloc[-1]
                    print(symbol, args.timerange, dtime.fromtimestamp(int(dataframe.iloc[1].otime) / 1000).strftime("%Y-%m-%d %H:%M:%S.%f") , dtime.fromtimestamp(int(dataframe.iloc[-1].otime) / 1000).strftime("%Y-%m-%d %H:%M:%S.%f"))


    def get_historical_klines(self, symbol, args):
        client =  Client()
        print('get_klines', symbol, args.timerange, args.timestart, 'START')
        start_time = time()

        untilThisDate = dtime.now()

        if self.tickers[symbol].get(args.timerange) != None:
            print(self.tickers[symbol][args.timerange])
            sinceThisDate = self.tickers[symbol][args.timerange]
        else:
            sinceThisDate = untilThisDate - timedelta(days=args.days)

        print('date start : ', sinceThisDate, ' date to :', untilThisDate)

        _klines = client.get_historical_klines(symbol, args.timerange, str(sinceThisDate), str(untilThisDate) )

        return symbol, _klines

    def get_klines(self, symbol, args):
        client =  Client()
        if args.days > 0:
            _klines = client.get_historical_klines(symbol, args.timerange, f'{args.days} day ago UTC' )
        if args.minutes > 0:
            _klines = client.get_historical_klines(symbol, args.timerange, f'{args.minutes} minutes ago UTC' )
        if args.hours > 0:
            _klines = client.get_historical_klines(symbol, args.timerange, f'{args.hours} minutes ago UTC' )
        return symbol, _klines




    def get_ticker_histo(self, stake,  ticker, timerange):
        dataframe = pd.DataFrame([], columns=['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore'])
        #print('data/'+ timerange)
        for filename in os.listdir('data/'+ timerange ) :
            if ticker+stake in filename:
                try:
                    df = pd.read_csv('data/'+ timerange +'/'+ filename)
                    dataframe = dataframe.append(df, ignore_index=True)
                    dataframe.sort_values('otime', inplace=True)
                    dataframe.drop_duplicates('otime', keep='last', inplace=True)
                finally:
                   #print(ticker, filename)
                    continue
            else:
                continue
        return dataframe


    def plot_display(self,  args):

        dataframe = self.get_ticker_histo(args.stake, args.ticker, args.timerange)
        dataframe = dataframe.apply(pd.to_numeric)
        dataframe['otime'] = pd.to_datetime( dataframe['otime'], unit='ms')
        dataframe = dataframe.set_index('otime')
        fig = go.Figure(data=[go.Candlestick(x=dataframe['otime'],
                                     open=dataframe['open'],
                                     high=dataframe['high'],
                                     low=dataframe['low'],
                                     close=dataframe['close'], name='market data')])

        # Show
        fig.show()

        return

    def plot_figure(self, fig,  variables, dataframe, args):
        dataframe = dataframe.apply(pd.to_numeric)
        dataframe['otime'] = pd.to_datetime( dataframe['otime'], unit='ms')
        #fig = go.Figure()

        fig = go.Figure(data=[go.Candlestick(x=dataframe['otime'],
                    open=dataframe['open'],
                    high=dataframe['high'],
                    low=dataframe['low'],
                    close=dataframe['close'],
                    name = "Candlestick"),
                    go.Scatter(x=dataframe['otime'], y=dataframe['pol_close'], line=dict(color='green', width=1)),
                    go.Scatter(x=dataframe['otime'], y=dataframe['pol_mean'], line=dict(color='red', width=1)),
                    #go.Scatter(mode='markers',
                    #    x=dataframe['otime'],
                    #    y=dataframe['buy'],
                    #    marker=dict(
                    #        color='LightSkyBlue',
                    #        size=20,
                    #        line=dict(
                    #            color='MediumPurple',
                    #            width=2
                    #        )
                    #    ),
                    #    showlegend=False
                    #)
                    ])
        fig.show()


    def plot_dataframe(self,  variables,  args):
        dataframe = dataframe.apply(pd.to_numeric)
        dataframe['otime'] = pd.to_datetime( dataframe['otime'], unit='ms')
        dataframe = dataframe.set_index('otime')

        fig, ax = plt.subplots()

        # andlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

        # Setting labels & titles
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        fig.suptitle('Wb Chart of ' + args.ticker)

        # Formatting Date
        date_format = mdates.DateFormatter('%d-%m-%Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        fig.tight_layout()

        x_data = variables['x_data']  # the index will be our x axis, not date
        y_data = variables['y_data']

        x = variables['x']
        y_pol = variables['y_pol']

        ax.plot(x_data, y_data, 'o', markersize=1.5, color='grey', alpha=0.7)

        # plot polynomial fit
        ax.plot(x, y_pol, '-', markersize=1.0, color='black', alpha=0.9)


        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend(loc=2)
        plt.title(args.ticker)

        plt.show()

    def plot_pol(self, args, x, y):
        fig, ax = plt.subplots(4, 4, figsize=(16, 16))
        for n in range(4):
            for k in range(4):
                order = 20 * n + 10 * k + 1
                #print(order)
                z = np.polyfit(x, y, order)
                p = np.poly1d(z)

                ax[n, k].scatter(x, y, label="Real data", s=1)
                ax[n, k].scatter(x,p(x), label="Polynomial with order={}".format(order), color='C1')
                ax[n, k].legend()
        fig.show()

    def plot_pol2(self, df, variables, args):
        fig, ax = plt.subplots(5, 5, figsize=(20, 20))
        order = 31
        for n in range(5):
            for k in range(5):
                last = -1 * ( n + (10 * k) + 1 )
                #print(order)

                #z = np.polyfit(x[0:last], y[0:last], 31)
                #p = np.poly1d(z)

                ax[n, k].scatter(variables['x_data'][0:last], variables['y_data'][0:last], label="Real data", s=1)
                #ax[n, k].scatter(variables['x-data'][0:last], variables['y-data-pivot'][0:last], label="Real data", s=1)

                ax[n, k].scatter(variables['x_data'][0:last], variables['pol'][0:last],
                                 label="Polynomial with order={}".format(order), color='C1')

                #ax[n, k].scatter(variables['x-data'][0:last],variables['pol-pivot'][0:last], label="Polynomial with order={}".format(order), color='C1')
                #ax[n, k].scatter(variables['x-data'][0:last],variables['pol-5'][0:last], label="Polynomial with order={}".format(order), color='C1')
                #ax[n, k].scatter(variables['x-data'][0:last],variables['pol-30'][0:last], label="Polynomial with order={}".format(order), color='C1')

                ax[n, k].legend()
        fig.show()

    def plot_pol3(self, df, variables, args):
        if len(variables['pol']) > 1:
            y_data = variables['y_data'][-1* len(variables['pol']):]
            x_data = variables['x_data'][0: len(y_data)]

            if len(x_data) > 1 and len(x_data) == len(y_data):
                #print('y_data', y_data[-1], 'pol', str(variables['y_pol'][-1]))
                fig, ax = plt.subplots()
                ax.scatter(x_data, y_data, label="Real data", s=1)
                ax.scatter(x_data, variables['pol'],
                                 label="Polynomial with order={}".format(31), color='C1')
                fig.show()


    def plot_pol_T(self, args, x, y):
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        for n in range(1):
            for k in range(1):
                order = 20 * n + 10 * k + 1
                #print(order)
                #z = np.polyfit(x, y, order)
                z = T.fit(x, y, 31)
                p = np.poly1d(z)
                ax[n, k].scatter(x, y, label="Real data", s=1)
                #ax[n, k].scatter(x, p(x), label="Polynomial with order={}".format(order), color='C1')
                ax[n, k].scatter(x, p(x), label="Polynomial with order={}".format(order), color='C1')

                ax[n, k].legend()
        fig.show()

    def backtest_process(self, args):
        #self.config['users'][self.user]['process']['trade_' + trade + '_id'] = 'running'
        self.get_histos(args)
        dataframe = self.get_ticker_histo(args.stake, args.ticker, args.timerange)


        # fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
        #                                     open=dataframe['open'],
        #                                     high=dataframe['high'],
        #                                     low=dataframe['low'],
        #                                     close=dataframe['close'], name='market data')])

        # Show

        df = pd.DataFrame([], columns=['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore'])
        variables = {}
        variables['onPosition'] = False
        variables['cash'] = 100
        variables['l_min'] = []
        variables['l_max'] = []
        variables['max_val'] = []
        variables['min_max_val'] = []
        variables['buy'] = []
        variables['sell'] = []
        variables['trend'] = ['NONE']
        variables['signal'] = ''
        variables['pol'] = []
        variables['SH'] = []
        variables['SL'] = []
        variables['profits'] = 0
        variables['cash'] = 100
        variables['started'] = False
        variables['lookback'] = []
        variables['clock'] = ''
        variables['trading_zone'] = []
        variables['x-data'] = []
        variables['x-data-time'] = []
        variables['pol'] = []
        variables['pol-pivot'] = []
        variables['pol-5'] = []
        variables['pol-10'] = []
        variables['pol-15'] = []
        variables['pol-20'] = []
        variables['pol-30'] = []
        variables['pol-60'] = []
        variables['pol_mean'] = []
        variables['action'] = ''

        pol = 0
        started = False
        fig, ax = plt.subplots()
        #timestart history
        #timestart
        for index, row in dataframe.iterrows():
            df = df.append(dataframe.iloc[index], ignore_index=True)

            last_candle = df.iloc[-1]
            #print('trading', dtime.fromtimestamp(last_candle['ctime'] / 1000), last_candle['close'])

            if len(df) > 3 :
                #variables['lookback'].append(last_candle_processed)

                if dtime.strptime(args.timestart, '%d-%m-%Y_%H:%M') <= dtime.fromtimestamp(last_candle['otime'] / 1000) and dtime.strptime(args.timestop, '%d-%m-%Y_%H:%M') > dtime.fromtimestamp(last_candle['otime'] / 1000):
                    if variables['started'] == False:
                        variables['past_min'] = df.low.min()
                        variables['past_max'] = df.high.max()

                    variables['started'] = True

                    variables['clock'] = dtime.fromtimestamp((int(df.iloc[-1].otime) / 1000)).strftime("%d/%m/%Y %H:%M")
                    #last_candle_processed = self.candle_process(variables, df, args)

                    last_candle_processed = self.candle_process_freq(variables, df, args)

                    print(variables['clock'], variables['signal'])

                    #print(dtime.fromtimestamp(int(df.iloc[-1].otime) / 1000).strftime("%d/%m/%Y %H:%M"), 'dataframe',
                    #      len(dataframe), df.iloc[-1], df.iloc[-2])
                    #print(dtime.fromtimestamp(last_candle['ctime'] / 1000), 'trading', len(min_max), len(l_min), len(l_max))

                    if variables['started'] == False:
                        variables['started'] = True

                    #if variables['started']:
                        #self.plot_pol3(df, variables, args)

                    if variables['signal'] == 'BUY' and variables['onPosition'] == False:
                        print('BUY', last_candle.close)
                        variables['buyPosition'] = last_candle.close
                        variables['onPosition'] = True
                        #self.plot_figure(fig, variables, df, args)
                        variables['signal'] = ''
                        #self.plot_pol3(df, variables, args)

                    if variables['signal'] == 'SELL' and variables['onPosition'] == True:
                        variables['profits'] += last_candle.close - variables['buyPosition']
                        print('SELL', last_candle.close, last_candle.close - last_candle.close, variables['profits'],
                              variables['cash'] + variables['cash'] * variables['profits'])
                        variables['onPosition'] = False
                        #self.plot_wb(df, fig, ax, args)
                        #self.plot_pol3(df, variables, args)

                    if variables['action'] == 'SELLING':
                        self.plot_pol3(df, variables, args)

                    if args.debug and dtime.strptime(args.pausestart, '%d-%m-%Y_%H:%M') <= dtime.fromtimestamp(last_candle['otime'] / 1000) and dtime.strptime(args.pausestop, '%d-%m-%Y_%H:%M') >= dtime.fromtimestamp(last_candle['otime'] / 1000) :
                        self.plot_figure(fig, variables, df, args)
                        #self.plot_wb(df, fig, ax, args)
                        #self.plot_fibo(df.iloc[-24:], fig, ax, variables['past_min'], variables['past_max'], args)
                        #self.plot_pol2(df, variables, args)
                        #self.plot_fibo(df.iloc[-24:], fig, ax, variables['past_min'], variables['past_max'], args)



                    #else:
                    #last_candle_processed = self.candle_process(variables, df, args)
                else:
                    last_candle_processed = self.candle_process(variables, df, args)
        #for each row
        #self.plot_wb(df, fig, ax, args)
        #self.plot_fibo(df.iloc[-9*60:], fig, ax, variables['past_min'], variables['past_max'], args)



    def candle_process(self, variables, dataframe, args):
        last_candle = dataframe.iloc[-1]
        x_data = dataframe.index.tolist()  # the index will be our x axis, not date
        y_data_low = dataframe['low']
        y_data_close= dataframe['close']
        y_data_pivot = (last_candle.high - last_candle.low) / 2


        # x values for the polynomial fit, 200 points

        x = np.linspace(0, max(dataframe.index.tolist()), max(dataframe.index.tolist()) + 1)
        variables['x-data-time'].append(dataframe['otime'].tolist())

        # polynomial fit of degree xx
        pol = np.polyfit(x_data, y_data_close, 21) #31)
        data = np.polyval(pol, x)
        #variables['pol'] = data

        #pol-pivot = np.polyfit(x_data, y_data_pivot, 21) #31)
        #data_pivot = np.polyval(pol, x)
        #variables['pol-pivot'] = data_pivot

        #           ___ detection of local minimums and maximums ___

        min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1  # local min & max
        l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
        l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max


        variables['x_data'] = x_data
        variables['y_data'] = y_data_close
        variables['x'] = x
        variables['y_pol'] = data


        #_pol = Decimal( str(data[-1]).split('.')[0] + '.' +
        #                str(data[-1]).split('.')[1][0:len(str(last_candle.close).split('.')[1])-2])

        pol_decimal = Decimal(str(data[-1]).split('.')[0] + '.' +
                str(data[-1]).split('.')[1][0:len(str(last_candle.close).split('.')[1]) - 2])
        variables['pol'].append(pol_decimal)
        dataframe['pol'] =  pol_decimal

        if len(dataframe) > 5:
            dataframe['pol_mean'] = dataframe['pol'].rolling(5).mean()
            variables['pol_mean'].append(dataframe['pol_mean'].iloc[-1])

        #if len(dataframe) > 60:
            #variables['pol-5'].append(variables['y_pol'][-5])#pol_T = T.fit(x_data, y_data, 31)
            #variables['pol-10'].append(variables['y_pol'][-10])
            #variables['pol-15'].append(variables['y_pol'][-15])
            #variables['pol-20'].append(variables['y_pol'][-20])
            #variables['pol-30'].append(variables['y_pol'][-30])  # pol_T = T.fit(x_data, y_data, 31)
            #variables['pol-60'].append(variables['y_pol'][-60])
        #if len(min_max) > 2:
        #    dataframe['valid'] = np.where( min_max[-1] - min_max[-2] > 10  ,1, 0)
        # if trend > 0 and trend(-1)  < 0 -> start  buying


        if variables['started'] == True and len(data) > 2 and len(variables['pol_mean']) > 2:
            if  data[-1] > data[-2]:
                variables['trend'].append('UP')

            if  data[-1] < data[-2]:
                # get the amount of bitcoin we have in our account
                variables['trend'].append('DOWN')


            if variables['pol_mean'][-1] >  variables['pol_mean'][-2]:
                variables['signal'] = 'BUY'
                variables['action'] = 'SELLING'


            if variables['pol_mean'][-1] <  variables['pol_mean'][-2]:
                variables['signal'] = 'SELL'
                variables['action'] = 'BUYING'


            #if len(dataframe) > 5 and self.plot_wb(df, fig, ax, args):
            #if variables['trend'][-1] == 'UP' and  last_candle.close < variables['pol'][-1]:
            #    variables['signal'] = 'BUY'


            #if variables['trend'][-1] == 'DOWN' and  variables['trend'][-2] == 'UP':
            #    variables['signal'] = 'SELL'


            #if variables['signal'] == 'BUYING' and variables['pol']['-1'] < variables['pol'][-2]:
            #    variables['signal'] = 'BUY'

            #if variables['signal'] == 'SELLING' and variables['pol']['-1'] > variables['pol'][-2]:
            #    variables['signal'] = 'SELL'

            # if variables.signal == 'BUYING' and variables.onPosition == False:
            #print(variables['trend'],dataframe.iloc[-1]['close'])
            # -1 100.00559
            # -4 100.01088

        return last_candle


    def backtest(self, args):
        args.ticker, args.stake = args.trade.split('/')
        self.backtest_process(args)
        return


    def _get_historical_data(self, symbol, timerange, args):

        # Create timedelta to get exact minute for end of historical data
        history_end = str(self.start_time.replace(second=0, microsecond=0) - timedelta(seconds=1))
        # Get historical klines up until current minute, truncating
        now = datetime.now()
        klines = self.client.get_historical_klines(symbol=symbol, interval=timerange,
                                                   start_str=self.history_start, end_str=history_end)
        print(f"Time taken to retrieve historical data:\n\t{datetime.now() - now}")
        hdf = pd.DataFrame(klines)
        hdf = hdf.iloc[:, :6]
        hdf[0] = pd.to_numeric(hdf[0], downcast='integer')
        hdf.iloc[:, 1:] = hdf.iloc[:, 1:].astype(float)
        hdf.columns = ['servertime', 'open', 'high', 'low', 'close', 'volume']
        # Compute timestamp
        hdf['kline_time'] = pd.to_datetime(hdf['servertime'], unit='ms')
        hdf['source'] = 'historical'

        return hdf[['servertime', 'kline_time', 'open', 'high', 'low', 'close', 'volume', 'source']]


    def bye(self, args):
        print('Dying ...')
        self.ps_stop(args)
        self.telegram._send_msg('*bye 4kings bot:*')
        self.telegram.cleanup()
        if (exchange.socket):
            print('Closing WebSocket connection...')
            exchange.close_socket()
        else:
            print('stopping strategy...')
            exchange.strategy.stop()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        x_data = dataframe.index.tolist()  # the index will be our x axis, not date

        y_data_close = dataframe['close']


        x = np.linspace(0, max(dataframe.index.tolist()), max(dataframe.index.tolist()) + 1)

        pol_close = np.polyfit(x_data, y_data_close, 21)
        dataframe['pol_close'] = np.polyval(pol_close, x)

        dataframe['pol_mean'] = dataframe['pol_close'].rolling(5).mean()

        #if len(min_max) > 2:
        #    dataframe['valid'] = np.where( min_max[-1] - min_max[-2] > 10  ,1, 0)
        # if trend > 0 and trend(-1)  < 0 -> start  buying
            # if start_buing and close < pol -> buy, a  action = action -1
            # if action =


        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
               (dataframe['pol_mean'] >  dataframe['pol_mean'].shift(1)) &
               (dataframe['close'] < dataframe['pol_close'])
               #(dataframe['close'] < dataframe['open']) &
               #(dataframe['close'] == dataframe['low']) &
               #(dataframe['open'] == dataframe['high']) &
               #(dataframe['open'] <  dataframe['pol-10'])

            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """

        dataframe.loc[
            (
                (dataframe['pol_mean'] < dataframe['pol_mean'].shift(1))
                #(dataframe['close'] > dataframe['open']) &
                ##(dataframe['close'] == dataframe['high']) &
                #(dataframe['open'] == dataframe['low']) &
                #(dataframe['open'] >  dataframe['pol-10'])
            ),
            'sell'] = 1
        return dataframe


    def candle_process_freq(self, variables, dataframe, args):
        last_candle = dataframe.iloc[-1]

        df = self.populate_indicators(dataframe, variables)

        df_buy = self.populate_buy_trend(df, variables)
        buy_values = df_buy['buy'].values
        df_sell = self.populate_sell_trend(df, variables)
        sell_values = df_buy['sell'].values

        mean_values = df['pol_mean'].values

        if buy_values[-1] is not None and buy_values[-1] == 1:
            variables['trend'].append('BUYING')

        if sell_values[-1] is not None and sell_values[-1] == 1:
            variables['trend'].append( 'SELLING' )

        if sell_values[-1] is None and buy_values[-1] is None:
            variables['trend'].append( variables['trend'][-1] )

        if len(mean_values) > 2 and variables['trend'][-1] == 'BUYING' and  mean_values[-1] > mean_values[-2]:
            variables['signal'] = 'BUY'
            variables['trend'][-1] = 'BUY'
            print('BUY', last_candle.close)

        if len(mean_values) > 2 and variables['trend'][-1] == 'SELLING' and  mean_values[-1] < mean_values[-2]:
            variables['signal'] = 'SELL'
            variables['trend'][-1] = 'SELL'
            print('SELL', last_candle.close)

        #df['buy_signal'] = np.where(df['buy' == 1, last_candle.close, nan])
        #df['sell_signal'] = np.where(df['sell' == 1, last_candle.close, nan])
        print('trend', variables['trend'][-1], mean_values[-1], variables['signal'] )
        return last_candle

def interactive_shell():
    """
    Like `interactive_shell`, but doing things manual.
    """
    batch = Trading()
    args = batch.parser.parse_args()
    batch.init(args)

    our_history = FileHistory('.4kings-history-file')

    print('args cmde', args)

    # Create Prompt.
    session = PromptSession('Say something: ', history=our_history)
    # Run echo loop. Read text from stdin, and reply it back.
    while True:
        try:
            result = session.prompt('>>', default='')
            #result = await PromptSession().prompt_async('>> ')
            print('You said: "{0}"'.format(result))
            if result != '':
                args = batch.parser.parse_args(result.split())
                args.cmde = result
                args.func(args)
                if args.cmde in 'bye':
                    print('GoodBye!')
                    return

        except Exception as _error:
            print(_error)
            pass


if __name__ == '__main__':
    interactive_shell()