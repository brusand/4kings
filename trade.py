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

class FourKings():
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

        self.parser_exchanges = self.subparsers.add_parser('exchange')
        self.parser_exchanges.add_argument('exchange', nargs='?',  action="store", default='*')
        self.parser_exchanges.add_argument('--apikey', help='apikey', required=False)
        self.parser_exchanges.add_argument('--secret', help='secret', required=False)
        self.parser_exchanges.add_argument('--stake', help='stake', required=False)
        self.parser_exchanges.add_argument('--asset', help='asset', required=False)
        self.parser_exchanges.add_argument('--balance', help='balance', action="store_true", default=False)
        self.parser_exchanges.set_defaults(func=self.exchanges)

        self.parser_tickers = self.subparsers.add_parser('ticker')
        self.parser_tickers.add_argument('ticker', nargs='?', action="store", default='*')
        self.parser_tickers.add_argument('--stake', nargs='?', help='stake', required=False)
        self.parser_tickers.add_argument('--load', action="store_true", default=False)
        self.parser_tickers.add_argument('--update', action="store_true", default=False)
        self.parser_tickers.add_argument('--list', action="store_true", default=False)
        self.parser_tickers.add_argument('--every', nargs='?', type=int,  action="store", default='60')
        self.parser_tickers.add_argument('--batch', action="store_true", default=False)
        self.parser_tickers.add_argument('--plot', action="store_true", default=False)
        self.parser_tickers.set_defaults(func=self.tickers)

        self.parser_histo = self.subparsers.add_parser('histo')
        self.parser_histo.add_argument('histo', nargs='?', action="store", default='*')
        self.parser_histo.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_histo.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_histo.add_argument('--timerange', help='time range', required=False, default='1m')
        self.parser_histo.add_argument('--days', help='days', type=int, action="store", required=False, default=0)
        self.parser_histo.add_argument('--hours', help='days', type=int, action="store", required=False, default=0)
        self.parser_histo.add_argument('--minutes', help='days', type=int, action="store", required=False, default=0)
        self.parser_histo.add_argument('--backtest', action="store_true", default=False)
        self.parser_histo.add_argument('--timestart', help='time start', required=False)
        self.parser_histo.add_argument('--timeto', help='time to', required=False)
        self.parser_histo.add_argument('--list', action="store_true", default=False)
        self.parser_histo.add_argument('--last', nargs='?', type=int,  action="store", required=False)
        self.parser_histo.add_argument('--print', nargs='?', type=int,  action="store", required=False)
        self.parser_histo.set_defaults(func=self.histos)

        self.parser_importer = self.subparsers.add_parser('importer')
        self.parser_importer.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_importer.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_importer.add_argument('--timerange', help='time range', required=False, default='1w')
        self.parser_importer.add_argument('--backtest', action="store_true", default=False)
        self.parser_importer.add_argument('--timestart', help='time start', required=False)
        self.parser_importer.add_argument('--timeto', help='time to', required=False)
        self.parser_importer.add_argument('--list', action="store_true", default=False)
        self.parser_importer.add_argument('--last', nargs='?', type=int,  action="store", required=False)
        self.parser_importer.add_argument('--print', nargs='?', type=int,  action="store", required=False)
        self.parser_importer.set_defaults(func=self.importer)


        self.parser_plot = self.subparsers.add_parser('plot')
        self.parser_plot.add_argument('plot', nargs='?', action="store", default='*')
        self.parser_plot.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_plot.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_plot.add_argument('--timerange', help='time range', required=False, default='1m')
        self.parser_plot.add_argument('--timestart', help='time start', required=False)
        self.parser_plot.add_argument('--timeto', help='time to', required=False)
        self.parser_plot.add_argument('--fibo', help='fibo', action="store_true", default=False)
        self.parser_plot.add_argument('--wb', help='wb', action="store_true", default=False)
        self.parser_plot.add_argument('--wb_fig', help='wb fig', action="store_true", default=False)
        self.parser_plot.add_argument('--wb_detect', help='wb_detect', action="store_true", default=False)
        self.parser_plot.add_argument('--close', help='close', action="store_true", default=False)
        self.parser_plot.set_defaults(func=self.plot)

        self.parser_zones = self.subparsers.add_parser('zones')
        self.parser_zones.add_argument('zones', nargs='?', action="store", default='*')
        self.parser_zones.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_zones.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_zones.add_argument('--trend', help='trend', required=False, default='FISRTCLASS')
        self.parser_zones.add_argument('--timerange', help='time range', required=False, default='1w')
        self.parser_zones.add_argument('--update', help='update', action="store_true", default=False)
        self.parser_zones.add_argument('--list', help='list', action="store_true", default=False)
        self.parser_zones.add_argument('--print', help='print', action="store_true", default=False)
        self.parser_zones.set_defaults(func=self.zones)

        self.parser_trade = self.subparsers.add_parser('trade')
        self.parser_trade.add_argument('trade', nargs='?', action="store", default='*')
        self.parser_trade.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_trade.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_trade.add_argument('--dry', help='dry', required=False, default=True)
        self.parser_trade.add_argument('--timerange', help='time range', required=False, default='1h')
        self.parser_trade.add_argument('--timestart', help='time start', required=False)
        self.parser_trade.add_argument('--timeto', help='time to', required=False)
        self.parser_trade.add_argument('--days', help='days', type=int, action="store", required=False, default=0)
        self.parser_trade.set_defaults(func=self.trade)

        self.parser_backtest = self.subparsers.add_parser('backtest')
        self.parser_backtest.add_argument('trade', nargs='?', action="store", default='ROSE/BNB')
        self.parser_backtest.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_backtest.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_backtest.add_argument('--dry', help='trend', required=False, default='FISRTCLASS')
        self.parser_backtest.add_argument('--timerange', help='time range', required=False, default='1m')
        self.parser_backtest.add_argument('--timestart', help='time start', required=False, default='21-01-2022_00:00')
        self.parser_backtest.add_argument('--timeto', help='time to', required=False)
        self.parser_backtest.add_argument('--debug', help='debug', action="store_true", default=False)
        self.parser_backtest.add_argument('--pausestart', help='time to', required=False, default='21-01-2022_01:30')
        self.parser_backtest.add_argument('--pausestop', help='time to', required=False, default='21-01-2022_01:50')
        self.parser_backtest.add_argument('--days', help='days', type=int, action="store", required=False, default=1)
        self.parser_backtest.add_argument('--hours', help='days', type=int, action="store", required=False, default=0)
        self.parser_backtest.add_argument('--minutes', help='days', type=int, action="store", required=False, default=0)
        self.parser_backtest.set_defaults(func=self.backtest)


        self.parser_ps = self.subparsers.add_parser('ps')
        self.parser_ps.add_argument('ps', nargs='?', action="store", default='')
        self.parser_ps.add_argument('--list', action="store_true", default=False)
        self.parser_ps.add_argument('--restart', action="store_true", default=False)
        self.parser_ps.add_argument('--stop', action="store_true", default=False)
        self.parser_ps.add_argument('--pop', nargs='?', action="store", default='')
        self.parser_ps.add_argument('--purge', action="store_true", default=False)
        self.parser_ps.set_defaults(func=self.ps)

        self.parser_bye = self.subparsers.add_parser('bye')
        self.parser_ps.add_argument('bye', nargs='?', action="store", default='')
        self.parser_bye.add_argument('--ps', nargs='?', action="store", default='*')
        self.parser_bye.add_argument('--now', action="store_true", default=False)
        self.parser_bye.add_argument('--force', action="store_true", default=False)
        self.parser_bye.set_defaults(func=self.bye)

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

    def ohlcv(dt, pair, period='11'):
        ohlcv = []
        limit = 1000
        if period == '1m':
            limit = 720
        elif period == '1d':
            limit = 365
        elif period == '1h':
            limit = 24
        elif period == '5m':
            limit = 288
        elif period == '1m':
            limit = 60;

        for i in dt:
            start_dt = datetime.strptime(i, "%Y%m%1w")
            since = calendar.timegm(start_dt.utctimetuple()) * 1000
            if period == '1m':
                ohlcv.extend(min_ohlcv(start_dt, pair, limit))
            else:
                ohlcv.extend(binance.fetch_ohlcv(symbol=pair, timeframe=period, since=since, limit=limit))
        df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Time'] = [dtime.fromtimestamp(float(time) / 1000) for time in df['Time']]
        df['Open'] = df['Open'].astype(np.float64)
        df['High'] = df['High'].astype(np.float64)
        df['Low'] = df['Low'].astype(np.float64)
        df['Close'] = df['Close'].astype(np.float64)
        df['Volume'] = df['Volume'].astype(np.float64)
        df.set_index('Time', inplace=True)
        return df

    def get_asset_balance(self, currency):
        client =  Client(self.config['users'][self.user]['key'], self.config['users'][self.user]['secret'])
        response = client.get_asset_balance(currency)
        return response['free']

    def order(self, order: Order):
        client =  Client(self.config['users'][self.user]['key'], self.config['users'][self.user]['secret'])
        return client.create_order(
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=order.quantity,
            price=order.price
        )

    def test_order(self, order: Order):
        client =  Client(self.config['users'][self.user]['key'], self.config['users'][self.user]['secret'])
        return client.create_test_order(
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=order.quantity,
            price=order.price
        )

    def check_order(self, orderId, symbol):
        client =  Client(self.config['users'][self.user]['key'], self.config['users'][self.user]['secret'])
        return client.get_order(
            symbol=symbol,
            orderId=orderId
        )

    def cancel_order(self, orderId, symbol):
        return self.client.cancel_order(
            symbol=symbol,
            orderId=orderId
        )


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

    def displayStakes(self, stake, args):
        print(stake)

    def displayTicker(self, ticker, args):
        print(ticker)

    def get_tickers(self, args):
        client = Client()
        info = client.get_exchange_info()
        symbols = [x['symbol'] for x in info['symbols']]
        exclude = ['DOWN', 'BEAR', 'BULL']  # leverage coins
        non_lev = [symbol for symbol in symbols if all(excludes not in symbol for excludes in exclude)]
        relevant = [symbol for symbol in non_lev if symbol.endswith(args.stake)]

        relevant.sort()
        return relevant

    def tickers_update(self, pid, value, args):
        # track new symbol
        self.ps_update(pid, 'waiting')
        waiting_time = True
        while waiting_time and not  self.config['users'][self.user]['process'][pid] in 'stop':
            try:
                self.ps_update(pid, 'running')
                tickers = self.get_tickers(args)
                for ticker in tickers:
                    if self.tickers.get(ticker) == None:
                        self.tickers[ticker] = {}
                        print('add new', ticker)
                        self.telegram._send_msg(f'*new ticker: {ticker}')
                        self.tickers.write()
                        print(args.stake, 'tickers updated')

            except (RuntimeError, TypeError, NameError):
                pass
            if args.every:
                self.ps_update(pid, 'waiting')
                sleep( args.every )
            else:
                waiting_time = False
        self.ps_update(pid, 'success')

    def tickers(self, args):
        if args.update:
            self.action_exec_args('tickers_update', '', args)
            #self.tickers_update(args)

        if args.load:
            self.tickers_load(args)

        if args.list:
            for ticker in self.tickers.keys():
                if args.ticker in '*' or args.ticker in ticker:
                    self.displayTicker(ticker, args)

    def get_local_tickers(self, args):
        tickers=[]
        for ticker in self.tickers.keys():
            if args.stake != '*':
                if args.stake in ticker:
                    if args.ticker == '*' or args.ticker in ticker:
                        tickers.append(ticker)
            else:
                if args.ticker == '*' or args.ticker in ticker:
                        tickers.append(ticker)
        return tickers

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

    def add_zone(self, sym, timerange, zone, close):
        if self.tickers[sym].get('zones') == None:
            self.tickers[sym]['zones'] = {}

        if self.tickers[sym]['zones'].get(timerange) == None:
            self.tickers[sym]['zones'][timerange] = ""

        if self.tickers[sym]['zones'].get(timerange + '_trend') == None:
            self.tickers[sym]['zones'][timerange + '_trend'] = '_2_' + zone
        else:
            self.tickers[sym]['zones'][timerange + '_trend'] = self.tickers[sym]['zones'][timerange] + '_2_' + zone
        self.tickers[sym]['zones'][timerange] = zone
        self.tickers[sym]['zones'][timerange + '_close'] = close
        print('updated', sym, self.tickers[sym]['zones'][timerange])

    def zones_print(self,  args):
        for ticker in self.get_local_tickers(args):
            if args.zones == '*' or self.tickers[ticker]['zones'][args.timerange] in args.zones and self.tickers[ticker]['zones'][args.timerange] != '':
                dataframe = self.get_ticker_histo(args.stake, ticker, args.timerange)
                zone = ''
                if len(dataframe) > 2:
                    retracements = self.fibo_retracements(dataframe['low'].min(), dataframe['high'].max())
                    last_closed_candle = dataframe.iloc[-2]

                    if last_closed_candle.close <= retracements[ECOCLASS_START] and last_closed_candle.close >= retracements[ECOCLASS_END]:
                        zone =  'ECOCLASS'

                    if last_closed_candle.close <= retracements[FIRSTCLASS_START] and last_closed_candle.close >= retracements[
                        FIRSTCLASS_END]:
                        zone =  'FIRSTCLASS'

                    if last_closed_candle.close < retracements[FIRSTCLASS_END]:
                        zone = 'DYING'

                    if last_closed_candle.close > retracements[ECOCLASS_START]:
                        zone = 'MISSEDTRAIN'

                    print(ticker, args.timerange, self.tickers[ticker]['zones'][args.timerange], self.tickers[ticker]['zones'][args.timerange+'_trend']) #, retracements)
        return

    def zone_update(self, sym, args):
        #print('fibo update', sym, args.stake)
        dataframe = self.get_ticker_histo(args.stake, sym, args.timerange)
        zone = ''
        if len(dataframe) > 2:
            retracements = self.fibo_retracements(dataframe['low'].min(), dataframe['high'].max())
            last_closed_candle = dataframe.iloc[-2]


            if last_closed_candle.close <= retracements[Lmax] and last_closed_candle.close >= retracements[L0114]:
                zone =  'ATH'

            if last_closed_candle.close <= retracements[L0114] and last_closed_candle.close >= retracements[L0214]:
                zone =  'TAKEPROFIT3'

            if last_closed_candle.close <= retracements[L0214] and last_closed_candle.close >= retracements[L0382]:
                zone =  'TAKEPROFIT2'

            if last_closed_candle.close <= retracements[L0382] and last_closed_candle.close >= retracements[L05]:
                zone =  'TAKEPROFIT'

            if last_closed_candle.close <= retracements[L05] and last_closed_candle.close >= retracements[L0618]:
                zone =  'MISSEDTRAIN'

            if last_closed_candle.close <= retracements[L0618] and last_closed_candle.close >= retracements[L0786]:
                zone =  'ECOCLASS'

            if last_closed_candle.close <= retracements[L0786] and last_closed_candle.close >= retracements[
                L0886]:
                zone =  'FIRSTCLASS'

            if last_closed_candle.close <= retracements[L0886] and last_closed_candle.close >= retracements[Lmin]:
                zone =  'SLEEPING'

            self.add_zone(sym, args.timerange, zone, last_closed_candle.close )
            self.tickers.write()

        return sym, zone


    def zones_list(self, args):
        for ticker in self.get_local_tickers(args):
            if args.zones == '*' or self.tickers[ticker]['zones'][args.timerange] in args.zones and self.tickers[ticker]['zones'][args.timerange] != '':
                if self.tickers[ticker]['zones'].get(args.timerange + '_trend') != None:
                    if  args.trend == '*' or args.trend in self.tickers[ticker]['zones'][args.timerange+'_trend']:
                        print(ticker, self.tickers[ticker]['zones'][args.timerange],  self.tickers[ticker]['zones'][args.timerange+'_trend'])

    def zones_updates(self, args):
        _zones={}
        relevant = [symbol for symbol in self.get_local_tickers(args)]
        callbacks = [lambda sym=symbol: self.zone_update(sym, args) for symbol in relevant]

        with ThreadPoolExecutor() as executor:
            tasks = [executor.submit(callback) for callback in callbacks]
            for task in tasks:
                symbol, zone = task.result()
                _zones[symbol] = zone

            for symbol in relevant:
                if self.tickers[symbol].get('zones') == None:
                    self.tickers[symbol]['zones'] = {}

                if self.tickers[symbol]['zones'].get(args.timerange+'_trend') != None:
                    print('zone', symbol, _zones[symbol], self.tickers[symbol]['zones'][args.timerange+'_trend'])
            self.tickers.write()
        return

    def fibo_retracements(self, min, max):
        diff = max - 0
        retracements = []
        retracements.append(max  - 0 * diff )
        retracements.append(max  - 0.114 * diff)
        retracements.append(max  - 0.214 * diff)
        retracements.append(max  - 0.382 * diff)
        retracements.append(max  - 0.5 * diff)
        retracements.append(max  - 0.618 * diff)
        retracements.append(max  - 0.786 * diff)
        retracements.append(max  - 0.886 * diff)
        retracements.append(max  - 1 * diff)
        return retracements

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

    def plot_wb_detect(self, ticker_df, args):
        #ticker_df =  self.get_ticker_histo(args.stake, args.ticker, args.timerange)

        #dataframe = dataframe.apply(pd.to_numeric)
        #dataframe['otime'] = pd.to_datetime( dataframe['otime'], unit='ms')
        #dataframe = dataframe.set_index('otime')
        #fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
        #                             open=dataframe['open'],
        #                             high=dataframe['high'],
        #                              low=dataframe['low'],
        #                             close=dataframe['close'], name='market data')])



        x_data = ticker_df.index.tolist()  # the index will be our x axis, not date
        y_data = ticker_df['close'] #low

        # x values for the polynomial fit, 200 points
        x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)

        # polynomial fit of degree xx
        pol = np.polyfit(x_data, y_data, 17)
        y_pol = np.polyval(pol, x)


        data = y_pol

        #           ___ detection of local minimums and maximums ___

        min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1  # local min & max
        l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
        l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max
        # +1 due to the fact that diff reduces the original index number

        # print('corresponding LOW values for suspected indeces: ')
        # print(ticker_df.Low.iloc[l_min])

        # extend the suspected x range:
        delta = 10  # how many ticks to the left and to the right from local minimum on x axis

        dict_i = dict()
        dict_x = dict()

        df_len = len(ticker_df.index)  # number of rows in dataset

        for element in l_min:  # x coordinates of suspected minimums
            l_bound = element - delta  # lower bound (left)
            u_bound = element + delta  # upper bound (right)
            x_range = range(l_bound, u_bound + 1)  # range of x positions where we SUSPECT to find a low
            dict_x[
                element] = x_range  # just helpful dictionary that holds suspected x ranges for further visualization strips

            # print('x_range: ', x_range)

            y_loc_list = list()
            for x_element in x_range:
                # print('-----------------')
                if x_element > 0 and x_element < df_len:  # need to stay within the dataframe
                    # y_loc_list.append(ticker_df.Low.iloc[x_element])   # list of suspected y values that can be a minimum
                    y_loc_list.append(ticker_df.low.iloc[x_element])
                    # print(y_loc_list)
                    # print('ticker_df.Low.iloc[x_element]', ticker_df.Low.iloc[x_element])
            dict_i[element] = y_loc_list  # key in element is suspected x position of minimum
            # to each suspected minimums we append the price values around that x position
            # so 40: [53.70000076293945, 53.93000030517578, 52.84000015258789, 53.290000915527344]
            # x position: [ 40$, 39$, 41$, 45$]
        # print('DICTIONARY for l_min: ', dict_i)

        # w detection
        y_delta = 0.12  # percentage distance between average lows
        threshold = min(ticker_df['low']) * 1.15  # setting threshold higher than the global low

        y_dict = dict()
        mini = list()
        suspected_bottoms = list()
        #   BUG somewhere here
        for key in dict_i.keys():  # for suspected minimum x position
            mn = sum(dict_i[key]) / len(
                dict_i[key])  # this is averaging out the price around that suspected minimum
            # if the range of days is too high the average will not make much sense

            price_min = min(dict_i[key])
            mini.append(price_min)  # lowest value for price around suspected

            l_y = mn * (1.0 - y_delta)  # these values are trying to get an U shape, but it is kinda useless
            u_y = mn * (1.0 + y_delta)
            y_dict[key] = [l_y, u_y, mn, price_min]

        # print('y_dict: ')
        # print(y_dict)

        # print('SCREENING FOR DOUBLE BOTTOM:')

        for key_i in y_dict.keys():
            for key_j in y_dict.keys():
                if (key_i != key_j) and (y_dict[key_i][3] < threshold):
                    suspected_bottoms.append(key_i)

        # ___ plotting ___
        plt.figure(figsize=(20, 10), dpi=120, facecolor='w', edgecolor='k')
        # plot stock data
        plt.plot(x_data, y_data, 'o', markersize=1.5, color='magenta', alpha=0.7)

        # we can plot also all the other prices to get a price range for given day just for information
        plt.plot(x_data, ticker_df['high'], 'o', markersize=1.5, color='blue', alpha=0.7)
        plt.plot(x_data, ticker_df['open'], 'o', markersize=1.5, color='grey', alpha=0.7)
        plt.plot(x_data, ticker_df['close'], 'o', markersize=1.5, color='red',
                 alpha=0.7)  # Adj Close should be more accurate indication (accounts for dividends and stock splits)
        #plt.plot(x_data, ticker_df['Adj close'], 'o', markersize=1.5, color='green', alpha=0.4)

        # plot polynomial fit
        plt.plot(x, y_pol, '-', markersize=1.0, color='black', alpha=0.9)

        #plt.plot(x[l_min], data[l_min], "o", label="min", color='r')  # minima
        #plt.plot(x[l_max], data[l_max], "o", label="max", color='b')  # maxima

        for position in suspected_bottoms:
            plt.axvline(x=position, linestyle='-.', color='r')

        plt.axhline(threshold, linestyle='--', color='b')

        for key in dict_x.keys():
            # print('dict key value: ', dict_i[key])
            for value in dict_x[key]:
                plt.axvline(x=value, linestyle='-', color='lightblue', alpha=0.2)

        plt.show()


    def plot_wb(self, ticker_df, fig, ax, args):
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

        x_data = ticker_df.index.tolist()  # the index will be our x axis, not date
        y_data = ticker_df['low']

        # x values for the polynomial fit, 200 points
        x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)

        # polynomial fit of degree xx
        pol = np.polyfit(x_data, y_data, 7)
        y_pol = np.polyval(pol, x)

        #           ___ plotting ___
        # plt.figure(figsize=(15, 2), dpi=120, facecolor='w', edgecolor='k')

        # plot stock data
        ax.plot(x_data, y_data, 'o', markersize=1.5, color='grey', alpha=0.7)

        # plot polynomial fit
        ax.plot(x, y_pol, '-', markersize=1.0, color='black', alpha=0.9)
        # ax.legend(['stock data', 'polynomial fit'])
        # plt.show()

        data = y_pol

        #           ___ detection of local minimums and maximums ___

        min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1  # local min & max
        l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
        l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max

        lmin = np.where(np.diff(np.sign(np.diff(data))) > 0, 1, 0)
        # +1 due to the fact that diff reduces the original index number

        # plot
        # plt.figure(figsize=(15, 2), dpi=120, facecolor='w', edgecolor='k')
        ax.plot(x, data, color='grey')
        ax.plot(x[l_min], data[l_min], "o", label="min", color='r')  # minima
        ax.plot(x[l_max], data[l_max], "o", label="max", color='b')  # maxima
        # ax.title('Local minima and maxima')

       # plt.ylabel("Price")
        #plt.xlabel("Date")
        #plt.legend(loc=2)
        #plt.title(args.ticker)

        fig.show()


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


    def plot_wb_fig(self, args):
        for ticker in self.get_local_tickers(args):
            if args.ticker == '*' or args.ticker in ticker:
                ticker_df =  self.get_ticker_histo(args.stake, ticker, args.timerange)

                fig, ax = plt.subplots()

                ticker_df = ticker_df.apply(pd.to_numeric)
                ticker_df['otime'] = pd.to_datetime( ticker_df['otime'], unit='ms')
                ticker_df = ticker_df.set_index('otime')
                fig = go.Figure(data=[go.Candlestick(x=ticker_df['otime'],
                                             open=ticker_df['open'],
                                             high=ticker_df['high'],
                                             low=ticker_df['low'],
                                             close=ticker_df['close'], name='market data')])


                #andlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

                # Setting labels & titles
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                fig.subtitle('Daily Candlestick Chart of ' + ticker)

                # Formatting Date
                date_format = mdates.DateFormatter('%d-%m-%Y')
                ax.xaxis.set_major_formatter(date_format)
                fig.autofmt_xdate()
                fig.tight_layout()

                ax.ylabel("Price")
                ax.xlabel("Date")
                ax.legend(loc=2)
                #ax.title(ticker)


                x_data = ticker_df.index.tolist()  # the index will be our x axis, not date
                y_data = ticker_df['low']

                # x values for the polynomial fit, 200 points
                x = np.linspace(0, max(ticker_df.index.tolist()), max(ticker_df.index.tolist()) + 1)

                # polynomial fit of degree xx
                pol = np.polyfit(x_data, y_data, 40)
                y_pol = np.polyval(pol, x)

                #           ___ plotting ___
                #plt.figure(figsize=(15, 2), dpi=120, facecolor='w', edgecolor='k')

                # plot stock data
                ax.plot(x_data, y_data, 'o', markersize=1.5, color='grey', alpha=0.7)

                # plot polynomial fit
                ax.plot(x, y_pol, '-', markersize=1.0, color='black', alpha=0.9)
                ax.legend(['stock data', 'polynomial fit'])
                #plt.show()

                data = y_pol

                #           ___ detection of local minimums and maximums ___

                min_max = np.diff(np.sign(np.diff(data))).nonzero()[0] + 1  # local min & max
                l_min = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1  # local min
                l_max = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1  # local max
                # +1 due to the fact that diff reduces the original index number

                # plot
                #plt.figure(figsize=(15, 2), dpi=120, facecolor='w', edgecolor='k')
                ax.plot(x, data, color='grey'),
                #ax.plot(x[l_min], data[l_min], "o", label="min", color='r')  # minima
                #ax.plot(x[l_max], data[l_max], "o", label="max", color='b')  # maxima
                ax.plot(x[l_max], dataframe['wb'], "o", label="wb", color='b')  # maxima
                #ax.title('Local minima and maxima')
                # Show


                fig.show()



    def plot_fibo(self, df, fig, ax, min, max, args):
        #andlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

        # Setting labels & titles
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        #fig.subtitle('Daily Candlestick Chart of ' + ticker)

        # Formatting Date
        date_format = mdates.DateFormatter('%d-%m-%Y')
        ax.xaxis.set_major_formatter(date_format)


        fig.tight_layout()

        ax.plot(df.close, color='black')

        price_min = min
        price_max = max

        # Fibonacci Levels considering original trend as upward move
        # Fibonacci Levels considering original trend as upward move
        diff = price_max - price_min

        level1_down = price_max - 0.114 * diff
        level2_down = price_max - 0.236 * diff
        level3_down = price_max - 0.382 * diff
        level4_down = price_max - 0.5 * diff
        level5_down = price_max - 0.618 * diff
        level6_down = price_max - 0.786 * diff
        level7_down = price_max - 0.886 * diff


        ax.axhspan(price_min, level7_down, alpha=0.4, color='lightsalmon')
        ax.axhspan(level7_down, level6_down, alpha=0.4, color='salmon')
        ax.axhspan(level6_down, level5_down, alpha=0.5, color='honeydew')
        ax.axhspan(level5_down, level4_down, alpha=0.5, color='grey')
        ax.axhspan(level4_down, level3_down, alpha=0.5, color='lightgreen')
        ax.axhspan(level3_down, level2_down, alpha=0.5, color='lime')
        ax.axhspan(level2_down, price_max, alpha=0.5, color='skyblue')

        #ax.hlines([level6_down_up, level6_down_down], 0, 50, transform=ax.get_yaxis_transform(), colors='r')
        fig.show()

    def zones(self, args):
        if args.update:
            self.zones_updates(args)
            return

        if args.print:
            self.zones_print(args)
            return

        if args.list:
            self.zones_list(args)
            return



    def histos(self, args):
        if args.histo != '*':
            args.ticker, args.stake = args.histo.split('/')

        if args.timestart != None:
            args.timestart = args.timestart.replace('_', '-')

        if args.timeto != None:
            args.timeto = args.timeto.replace('_','-')

        if args.timerange != None:
            if not os.path.exists('data/'+ args.timerange):
                os.makedirs('data/'+ args.timerange)

        if args.print:
            self.print_ticker_histo(args)
            return

        if args.last:
            self.print_histos(args)
            return
        else:
            self.get_histos(args)

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

    def plot(self, args):
        ticker, stake  = args.plot.split('/')
        args.stake = stake
        args.ticker = ticker

        if args.timestart != None:
            args.timestart = args.timestart.replace('_', '-')

        if args.timeto != None:
            args.timeto = args.timeto.replace('_','-')

        if args.fibo:
            self.plot_fibo(args)
            return

        if args.wb:
            self.plot_wb(args)
            return

        if args.wb_fig:
            self.plot_wb_fig(args)
            return

        if args.pol:
            self.plot_pol(args)
            return
        if args.wb_detect:
            self.plot_wb_detect(args)
            return

        self.plot_display(args)

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

    def detect_wb(self, zone):

        print(zone)
        second_leg =  zone.iloc[-2:]['close'].max()
        second_bottom = zone.iloc[-5:]['close'].min()
        middle = zone.iloc[-7:-3]['close'].max()
        first_bottom = zone.iloc[-10:-5]['close'].min()
        first_leg = zone.iloc[-12:-8]['open'].max()
        #print(first_leg, first_bottom, middle, second_bottom, second_leg)
        if second_leg > middle and \
            middle < second_leg and middle < first_leg and \
            middle > second_bottom and \
            middle  > first_bottom and \
            first_leg > middle:
                print('DETECTED', first_leg, first_bottom, middle, second_bottom, second_leg)
                return True

        return False

    def trade_process(self, args):
        #self.config['users'][self.user]['process']['trade_' + trade + '_id'] = 'running'

        if args.trade != '*':
            args.ticker, args.stake = args.trade.split('/')
        # get historic


        # display graph
        while true:
            self.get_histos(args)
            dataframe = self.get_ticker_histo(args.ticker + args.stake, args)
            last_candle = dataframe.iloc[-1]
            variables = {}
            variables.onPosition = False
            variables.cash = 100
            #fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
            #                                     open=dataframe['open'],
            #                                     high=dataframe['high'],
            #                                     low=dataframe['low'],
            #                                     close=dataframe['close'], name='market data')])

            # Show
            #fig.show()

            last_candle  = self.candle_process(variables, dataframe)


            if variables.signal == 'BUYING' and variables.onPosition == False:
                print('BUY', last_candle.close)
                variables.buyPosition = last_candle.close
                variables.onPosition = True

            if variables.signal == 'SELLING' and variables.onPosition == True:
                variables.profits += last_candle.close - variables.buyPosition
                print('SELL', last_candle.close, last_candle.close - last_candle.close, variables.profits, variables.cash * variables.profits)



            next_time = datetime.fromtimestamp(last_candle['ctime'] / 1000)
            while datetime.now() <= next_time:
                sleep(2)
                #if self.config['users'][self.user]['process'][trade_id] == 'stop':
                #    self.ps_update(process_id, 'stopped')
                #    return

        # wait for close time of last candle
            # get last closed candle
            # calculate indicator
            # enter in position if not already
            # exit position if already in
        return


    def trade(self, args):
        args.ticker, args.stake = args.trade.split('/')
        self.trade_process(args)
        return

    def backtest(self, args):
        args.ticker, args.stake = args.trade.split('/')
        self.backtest_process(args)
        return

    def exchange(self, args):
        return

    def strategie(self, args):
        return



    def exchanges(self, args):
        if args.exchange != None and args.exchange != '*':
            self.config['users'][self.user]['exchange'] = args.exchange

        if args.key != None and args.key != '*':
            self.config['users'][self.user]['key'] = args.key

        if args.secret != None and args.secret != '*':
            self.config['users'][self.user]['secret'] = args.secret

        if args.stake != None:
            self.config['users'][self.user]['stake'] = args.stake

        if args.asset != None:
            self.config['users'][self.user]['asset'] = args.asset

        if args.balance:
            print(self.config['users'][self.user]['stake'])
            balance = self.get_asset_balance(self.config['users'][self.user]['stake'])
            print('balance', balance)


    def importer(self, args):
        period_start = config('PERIOD_START')
        period_end = config('PERIOD_END')

        print(
            "Import mode on {} symbol for period from {} to {} with {} seconds candlesticks.".format(
                exchange.get_symbol(),
                period_start,
                period_end,
                interval
            )
        )
        importer = Importer(exchange, period_start, period_end, interval)
        importer.process()
        return

    def print_ticker_histo(self, args):
        dataframe = self.get_ticker_histo(args.stake, args.ticker, args.timerange)
        if dataframe.size > 0:
            for i in range(int(args.print), 0):
                candle = dataframe.iloc[i]
                print(dtime.fromtimestamp(int(candle.otime)/1000).strftime("%d/%m/%Y %H:%M"), dtime.fromtimestamp(int(candle.ctime)/1000).strftime("%d/%m/%Y %H:%M"), candle.open, candle.close, candle.high, candle.low)


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

    def print_histos(self, args):
        klines={}
        relevant = [symbol for symbol in self.get_local_tickers(args)]
        dt = [args.timestart, args.timeto]
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
                    print(symbol, args.timerange, ' to csv')
                    dataframe.columns = ['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore']
                    for i in range(int(args.last), 0):
                        candle = dataframe.iloc[i]
                        print(dtime.fromtimestamp(int(candle.otime)/1000).strftime("%d/%m/%Y %H:%M"), dtime.fromtimestamp(int(candle.ctime)/1000).strftime("%d/%m/%Y %H:%M"), candle.open, candle.close, candle.high, candle.low)


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


    def tickers_load(self, args):
        tickers = self.get_tickers(args)
        for ticker in tickers:
            if self.tickers.get(ticker) == None:
                self.tickers[ticker] = {}
                print('add', ticker)
        self.tickers.write()
        print(len(tickers), 'added')

        return tickers


    def ps(self, args):
        if args.pop:
            self.ps_pop(args.pop)

        if args.purge:
            self.ps_purge(args)

        if args.restart:
            self.ps_restart(args)

        if args.stop:
            self.ps_stop(args)

        if args.list:
            self.ps_list()

    def ps_pop(self, p_id):
        for process_id in self.config['users'][self.user]['process'].keys():
            if p_id in process_id:
                self.config['users'][self.user]['process'].pop(process_id)
                if self.config['users'][self.user].get('cmdes') is not None and self.config['users'][self.user].get(process_id) is not None:
                    self.config['users'][self.user]['cmdes'].pop(process_id)
                self.config.write()
                print ("process : ", process_id, "killed")

    def ps_stop(self, args):
        for process_id in self.config['users'][self.user]['process'].keys():
            if args.ps in process_id or args.ps in'*':
                self.ps_update(process_id, 'stop')

    def ps_update(self, process_id, status):
        self.config['users'][self.user]['process'][process_id] = status
        print ("process : ", process_id, status)

    def ps_list(self):
        for process_id in self.config['users'][self.user]['process'].keys():
            print ("process id  : ", process_id, "process name", self.config['users'][self.user]['process'][process_id])

    def ps_restart(self, args):
        if self.config['users'][self.user]['process'].keys is not None:
            for process_id in self.config['users'][self.user]['process'].keys():
                if not self.config['users'][self.user]['process'][process_id] in 'success':
                    args = self.parser.parse_args(self.config['users'][self.user]['cmdes'][process_id].split())
                    args.cmde = self.config['users'][self.user]['cmdes'][process_id]
                    #if args.cha is not None:
                    #    args.cha = args.cha.replace('_', '-')
                    args.func(args)
                self.ps_pop(process_id)

    def ps_purge(self, args):
        for process_id in self.config['users'][self.user]['process'].keys():
            self.ps_pop(process_id)

    def ps_add(self, process_id,  status, action, value, args):
        if self.config['users'][self.user].get('process') == None:
            self.config['users'][self.user]['process']= {}

        self.config['users'][self.user]['process'][process_id] = status

        self.cmde_add(process_id, action, value, args)

        if args.batch == True:
            self.config.write()

        print ('process', process_id, status)

    def cmde_list(self, process_id, status, args):
        for process_id in self.config['users'][self.user]['cmdes'].keys():
            print (self.config['users'][self.user]['cmdes'][process_id])

    def cmde_add(self, process_id, action, value, args):
        if self.config['users'][self.user].get('cmdes') == None:
            self.config['users'][self.user]['cmdes']= {}

        if self.config['users'][self.user]['cmdes'].get(process_id) == None:
            self.config['users'][self.user]['cmdes'][process_id] = args.cmde

            if args.batch == True:
                self.config.write()
            #print 'cmde', args.cmde
        else:
            print ('cmde', process_id , 'already exist')

    def action_exec_args(self, action, value, args):
        process_id = action + '-' + str(value) + '-'
        process_id += dtime.now().strftime('%Y-%m-%d_%H:%M')
        process_state = 'init'

        self.ps_add(process_id, process_state, action, value, args)

        self.threads[process_id] =  threading.Thread(target=getattr(self, action), name=action+str(value), kwargs=dict(pid=process_id, value=str(value), args=args))
        self.threads[process_id].daemon = True  # Daemonize thread
        self.threads[process_id].start()

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

    def analyze_last_candle(self, dataframe, args):
        return

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
    batch = FourKings()
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