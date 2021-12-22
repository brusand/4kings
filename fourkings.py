from __future__ import unicode_literals
import asyncio
import aiohttp
import argparse
from configobj import ConfigObj
from binance.client import Client
from datetime import datetime as dtime, time, timedelta
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

import pandas as pd
from time import sleep
from binance import AsyncClient, Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from concurrent.futures import ThreadPoolExecutor

from Telegram import Telegram;


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

        self.parser_tickers = self.subparsers.add_parser('ticker')
        self.parser_tickers.add_argument('ticker', nargs='?', action="store", default='*')
        self.parser_tickers.add_argument('--stake', nargs='?', help='stake', required=False)
        self.parser_tickers.add_argument('--load', action="store_true", default=False)
        self.parser_tickers.add_argument('--update', action="store_true", default=False)
        self.parser_tickers.add_argument('--list', action="store_true", default=False)
        self.parser_tickers.add_argument('--every', nargs='?', type=int,  action="store", default='60')
        self.parser_tickers.set_defaults(func=self.tickers)

        self.parser_histo = self.subparsers.add_parser('histo')
        self.parser_histo.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_histo.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_histo.add_argument('--timerange', help='time range', required=False, default='1w')
        self.parser_histo.add_argument('--backtest', action="store_true", default=False)
        self.parser_histo.add_argument('--timestart', help='time start', required=False)
        self.parser_histo.add_argument('--timeto', help='time to', required=False)
        self.parser_histo.add_argument('--list', action="store_true", default=False)
        self.parser_histo.add_argument('--last', nargs='?', type=int,  action="store", required=False)
        self.parser_histo.add_argument('--print', nargs='?', type=int,  action="store", required=False)
        self.parser_histo.set_defaults(func=self.histos)

        self.parser_zones = self.subparsers.add_parser('zones')
        self.parser_zones.add_argument('--stake', help='stake', required=False, default='*')
        self.parser_zones.add_argument('--ticker', help='ticker', required=False, default='*')
        self.parser_zones.add_argument('--timerange', help='time range', required=False, default='1w')
        self.parser_zones.add_argument('--update', help='update', action="store_true", default=False)
        self.parser_zones.set_defaults(func=self.zones)


        self.parser_ps = self.subparsers.add_parser('ps')
        self.parser_ps.add_argument('ps', nargs='?', action="store", default='')
        self.parser_ps.add_argument('--list', action="store_true", default=False)
        self.parser_ps.add_argument('--restart', action="store_true", default=False)
        self.parser_ps.add_argument('--stop', action="store_true", default=False)
        self.parser_ps.add_argument('--pop', nargs='?', action="store", default='')
        self.parser_ps.add_argument('--purge', action="store_true", default=False)
        self.parser_ps.set_defaults(func=self.ps)

        self.config = ConfigObj('./config/4kings.ini')
        if self.config.get('user') == None:
            self.config['users'] = {}

        else:
            self.tickers = ConfigObj('./config/tickers-' + self.config.get('user') + '.ini')
            self.tickers.write()

        self.config.write()
        self.telegram = Telegram( self.config )
        self.binance = ccxt.binance()


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

    def ohlcv(dt, pair, period='1d'):
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


    def get_klines(self, symbol, args):
        client =  Client()
        print('get_klines', symbol, args.timerange, args.timestart, 'START')
        if args.timestart == None:
            if self.tickers[symbol].get(args.timerange) != None:
                print(self.tickers[symbol][args.timerange])
                args.timestart = dtime.strptime(self.tickers[symbol][args.timerange], "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d")
            else:
                args.timestart = dtime(2021,1,1).strftime("%Y-%m-%d")
        else:
            tstart = args.timestart.split('-')
            print(tstart)
            args.timestart = dtime(int(tstart[0]), int(tstart[1]), int(tstart[2])).strftime("%Y-%m-%d")

        if args.timeto == None:
            args.timeto =  dtime.today().strftime("%Y-%m-%d")

        print('date start : ', args.timestart, ' date to :', args.timeto)
        _timestart = dtime.strptime(args.timestart, "%Y-%m-%d")
        _klines = client.get_historical_klines(symbol, interval=args.timerange, start_str=_timestart.strftime("%d %b %Y"))


        print('get_klines', symbol, args.timerange, args.timestart, 'END')
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

    def get_ticker_histo(self, ticker, timerange):
        dataframe = pd.DataFrame([], columns=['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore'])
        print('data/'+ timerange)
        for filename in os.listdir('data/'+ timerange):
            if ticker in filename:
                try:
                    df = pd.read_csv('data/'+ timerange+'/'+filename)
                    dataframe = dataframe.append(df, ignore_index=True)
                    dataframe.sort_values('otime', inplace=True)
                    dataframe.drop_duplicates('otime', keep='last', inplace=True)
                finally:
                    print(ticker, filename)
            else:
                continue
        return dataframe

    def zone_update(self, sym, args):
        #print('fibo update', sym, args.stake)
        dataframe = self.get_ticker_histo(sym, args.timerange)
        zone = 'neutre'
        #print('updated', sym, zone)
        return sym, zone

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
                self.tickers[symbol]['zones'][args.timerange] = _zones[symbol]
                print('zone', symbol, _zones[symbol])
            self.tickers.write()
        return

    def zones(self, args):
        if args.update:
            self.zones_updates(args)
            return

        if args.list:
            return

    def histos(self, args):
        if args.timestart != None:
            args.timestart = args.timestart.replace('_', ' ')

        if args.print:
            self.print_ticker_histo(args)
            return

        if args.last:
            self.print_histos(args)
            return
        else:
            self.get_histos(args)

    def print_ticker_histo(self, args):
        dataframe = self.get_ticker_histo(args.ticker, args.timerange)
        if dataframe.size > 0:
            for i in range(int(args.print), 0):
                candle = dataframe.iloc[i]
                print(dtime.fromtimestamp(int(candle.otime)/1000).strftime("%d/%m/%Y %H:%M"), dtime.fromtimestamp(int(candle.ctime)/1000).strftime("%d/%m/%Y %H:%M"), candle.open, candle.close, candle.high, candle.low)


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
        relevant = [symbol for symbol in self.get_local_tickers(args)]

        dt = [args.timestart, args.timeto]
        #callbacks = [lambda sym=symbol: self.get_klines(sym, args) for symbol in relevant]
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

                    if args.backtest:
                        dataframe.to_csv('./backtest/' + args.timerange + '/' +symbol + '_' + args.timestart + '_' + args.timeto + '.csv',
                                         encoding='utf-8',
                                         index=False)
                    else:
                        dataframe.to_csv('./data/' + args.timerange + '/' + symbol + '_' + args.timestart  + '_' + args.timeto + '.csv',
                                         encoding='utf-8',
                                         index=False)
                    candle = dataframe.iloc[-1]
                    self.tickers[symbol][args.timerange] = dtime.fromtimestamp(int(candle.otime) / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")

            self.tickers.write()

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
        self.config.write()
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
        self.config.write()
        self.cmde_add(process_id, action, value, args)
        print ('process', process_id, status)

    def cmde_list(self, process_id, status, args):
        for process_id in self.config['users'][self.user]['cmdes'].keys():
            print (self.config['users'][self.user]['cmdes'][process_id])

    def cmde_add(self, process_id, action, value, args):
        if self.config['users'][self.user].get('cmdes') == None:
            self.config['users'][self.user]['cmdes']= {}
        if self.config['users'][self.user]['cmdes'].get(process_id) == None:
            self.config['users'][self.user]['cmdes'][process_id] = args.cmde
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


    def cleanup(self):
        print('Dying ...')
        self.telegram._send_msg('*bye 4kings bot:*')
        self.telegram.cleanup()

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
                if 'bye' in result:
                    batch.cleanup()
                    print('GoodBye!')
                    return
                else:
                    args = batch.parser.parse_args(result.split())
                    args.cmde = result
                    args.func(args)
        except Exception as _error:
            print(_error)
            pass


if __name__ == '__main__':
    interactive_shell()