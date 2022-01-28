#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import os.path

import ipywidgets as widgets
import bqplot as bq
import bqplot.pyplot as plt
from bqplot import OrdinalScale
import datetime as dt
from datetime import datetime


# In[ ]:





# In[2]:


stake = 'BNB'
ticker = 'ROSE'
timerange = '1m'
time_interval = 100
rolling = 0


# In[3]:


def get_ticker_histo(stake,  ticker, timerange):
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


# In[ ]:





# In[4]:


data = get_ticker_histo(stake, ticker, timerange)


df = pd.DataFrame([], columns=['otime', 'open', 'high', 'low', 'close', 'volume', 'ctime', 'quote',
                                         'trades', 'TB Base Volume', 'TB Quote Volume', 'ignore'])
df = data.iloc[:1]


# In[5]:


timeframe_slider = widgets.IntSlider(
    min=1, max=len(data), description="timeframe", value=1
)


# In[6]:


#wealth_scat.x, wealth_scat.y, wealth_scat.size = get_data(year_slider.value)
#declare figure

from bqplot import OrdinalScale

fig = plt.figure()
plt.scales(scales={"x": OrdinalScale()})
axes_options = {
"x": {"label": "X", "tick_format": "%d-%m-%Y %H:%M"},
"y": {"label": "Y", "tick_format": ".2f"},
}

candle_stick = plt.ohlc(
df['otime'].index,
df[["open","high","low","close"]],
stroke_width=1.5, stroke="black", padding=0.2,
label_display=True, label_display_vertical_offset=-20,
label_font_style={"font-weight":"bold", "font-size":"15px", "fill": "white"},
)




# In[7]:


def timeframe_changed(change):
    #wealth_scat.x, wealth_scat.y, wealth_scat.size = get_data(year_slider.value)
    #year_label.text = [str(year_slider.value)]
    print(timeframe_slider)
    if rolling == 0 or timeframe_slider.value < rolling :
        start = 1
    else:
        start = timeframe_slider.value - rolling
    
    df  = data.iloc[start:timeframe_slider.value]

    candle_stick.x = df.index
    candle_stick.y = df[["open", "high", "low", "close"]]



# In[8]:


timeframe_slider.observe(timeframe_changed, "value")


# In[9]:


play_button = widgets.Play(min=1, max=len(data), interval=100)
widgets.jslink((play_button, "value"), (timeframe_slider, "value"))
widgets.VBox([widgets.HBox([play_button, timeframe_slider]), fig])

print('fin')
# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




