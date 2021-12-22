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


def get_ticker_histo(ticker, timeframe):
    dataframe = pd.DataFrame([], columns=['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                          'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore'])
    print('data/' + timeframe)
    for filename in os.listdir('data/' + timeframe):
        if ticker in filename:
            try:
                df = pd.read_csv('data/' + timeframe + '/' + filename)
                print(df)
                dataframe = dataframe.append(df, ignore_index = True)
                print(dataframe)
                # sorting by first name
                dataframe.sort_values('otime',  inplace=True)
                dataframe.drop_duplicates('otime', keep='last', inplace=True)
                print(dataframe)
            finally:
                print(ticker, filename)
        else:
            continue
    return dataframe


get_ticker_histo('SUPERUSDT', '1w')