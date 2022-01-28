#!/usr/bin/env python
# coding: utf-8

# This is a `bqplot` recreation of Mike Bostock's [Wealth of Nations](https://bost.ocks.org/mike/nations/). This was also done by [Gapminder](http://www.gapminder.org/world/#$majorMode=chart$is;shi=t;ly=2003;lb=f;il=t;fs=11;al=30;stl=t;st=t;nsl=t;se=t$wst;tts=C$ts;sp=5.59290322580644;ti=2013$zpv;v=0$inc_x;mmid=XCOORDS;iid=phAwcNAVuyj1jiMAkmq1iMg;by=ind$inc_y;mmid=YCOORDS;iid=phAwcNAVuyj2tPLxKvvnNPA;by=ind$inc_s;uniValue=8.21;iid=phAwcNAVuyj0XOoBL_n5tAQ;by=ind$inc_c;uniValue=255;gid=CATID0;by=grp$map_x;scale=log;dataMin=194;dataMax=96846$map_y;scale=lin;dataMin=23;dataMax=86$map_s;sma=49;smi=2.65$cd;bd=0$inds=;modified=60). It is originally based on a TED Talk by [Hans Rosling](http://www.ted.com/talks/hans_rosling_shows_the_best_stats_you_ve_ever_seen).

# In[1]:


import numpy as np
import pandas as pd

import bqplot as bq
import bqplot.pyplot as plt





def get_data(year):
    year_index = year - 1800
    income = data["income"].apply(lambda x: x[year_index])
    life_exp = data["lifeExpectancy"].apply(lambda x: x[year_index])
    pop = data["population"].apply(lambda x: x[year_index])
    return income, life_exp, pop


# In[8]:


time_interval = 100
fig_layout = widgets.Layout(width="1000px", height="700px", overflow_x="hidden")
fig = plt.figure(
    layout=fig_layout,
    fig_margin=dict(top=60, bottom=80, left=40, right=20),
    title="4 Kings strategy",
    animation_duration=time_interval,
)

plt.scales(
    scales={
        "x": bq.LogScale(min=min(200, income_min), max=income_max),
        "y": bq.LinearScale(min=life_exp_min, max=life_exp_max),
        "color": bq.OrdinalColorScale(
            domain=data["region"].unique().tolist(), colors=bq.CATEGORY10[:6]
        ),
        "size": bq.LinearScale(min=pop_min, max=pop_max),
    }
)

# add custom x tick values
ticks = [2, 4, 6, 8, 10]
income_ticks = (
    [t * 100 for t in ticks] + [t * 1000 for t in ticks] + [t * 10000 for t in ticks]
)

# custom axis options
axes_options = {
    "x": dict(
        label="Income per Capita",
        label_location="end",
        label_offset="-2ex",
        tick_format="~s",
        tick_values=income_ticks,
    ),
    "y": dict(
        label="Life Expectancy",
        orientation="vertical",
        side="left",
        label_location="end",
        label_offset="-1em",
    ),
    "color": dict(label="Region"),
}

tooltip = bq.Tooltip(
    fields=["name", "x", "y"],
    labels=["Country Name", "Income per Capita", "Life Expectancy"],
)

year_label = bq.Label(
    x=[0.75],
    y=[0.10],
    default_size=46,
    font_weight="bolder",
    colors=["orange"],
    text=[str(initial_year)],
    enable_move=True,
)

# Start with the first year's data
cap_income, life_exp, pop = get_data(initial_year)

wealth_scat = plt.scatter(
    cap_income,
    life_exp,
    color=data["region"],
    size=pop,
    names=data["name"],
    display_names=False,
    default_size=20000,
    tooltip=tooltip,
    stroke="Black",
    axes_options=axes_options,
    unhovered_style={"opacity": 0.5},
)
nation_line = plt.plot(
    data["income"][0], data["lifeExpectancy"][0], colors=["Gray"], visible=False
)

# slider for the year
year_slider = widgets.IntSlider(
    min=1800, max=2008, description="Year", value=initial_year
)

# register callbacks
def hover_changed(change):
    if change.new is not None:
        nation_line.x = data[data["name"] == wealth_scat.names[change.new]][
            "income"
        ].values[0]
        nation_line.y = data[data["name"] == wealth_scat.names[change.new]][
            "lifeExpectancy"
        ].values[0]
        nation_line.visible = True
    else:
        nation_line.visible = False


wealth_scat.observe(hover_changed, "hovered_point")


def year_changed(change):
    wealth_scat.x, wealth_scat.y, wealth_scat.size = get_data(year_slider.value)
    year_label.text = [str(year_slider.value)]


year_slider.observe(year_changed, "value")

play_button = widgets.Play(min=1800, max=2008, interval=time_interval)
widgets.jslink((play_button, "value"), (year_slider, "value"))

widgets.VBox([widgets.HBox([play_button, year_slider]), fig])


# In[ ]:




