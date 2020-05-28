# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:12:34 2020

@author: tsaof
"""
## Import and setup required packages
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

## Set up css framework
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
## Read in cleaned up stock data
stock_data = pd.read_csv('stock_data.csv')

## Set up table of portfolio to be shown as a Plotly object
data_table = stock_data[['Ticker','Purchase Date', 'Unit Cost', 'Quantity', 'Present Price']]
trace_portfolio = go.Table(header = dict(values = [x for x in data_table.columns],
                                         fill_color = '#2E93CE',
                                         line_color = 'white',
                                         align = 'center',
                                         font=dict(color='white', size=16)),
                       cells = dict(values = np.array(data_table.values.tolist()).T,
                                    fill_color = '#2E93CE',
                                    line_color = 'white',
                                    align = 'center',
                                    font=dict(color='white', size=14)))
table_portfolio = [trace_portfolio]
layout_table_portfolio = dict(title = "Portfolio Composition",
                              title_x = 0.5,
                              titlefont=dict(
                                  family='sans-serif',
                                  size=28,
                                  color="#0A537D")
                              )
fig_2 = dict(data = table_portfolio,
             layout = layout_table_portfolio,
             )

## Set up bar chart of portfolio vs. Sp500 performance
comparison_trace1 = go.Bar(y=stock_data['Stock Gain'], x=stock_data['Ticker'],
           name='Portfolio', width=0.4,
           marker_color='#2E93CE', marker_line_color='#2E93CE'
           )
comparison_trace2 = go.Bar(y=stock_data['SP500 Gain'], x=stock_data['Ticker'],
           name='SP500', width=0.4,
           marker_color='#FFB266', marker_line_color='#FFB266'
           )
bar_comparison = [comparison_trace1, comparison_trace2]
layout_bar_comparison = dict(title = 'Portfolio vs. SP500 Gain/Loss',
                             title_x = 0.5,
                             titlefont=dict(
                                 family='sans-serif',
                                 size=28,
                                 color="#0A537D")
                             )
fig_3 = dict(data = bar_comparison, layout = layout_bar_comparison)

## Set up time-series monitoring group
# Specify drop-down menu options for stock monitoring graph
tickers = [ticker for ticker in stock_data.Ticker]
option_list = []
for ticker in tickers:
	dict_ticker = {}
	dict_ticker['label'] = ticker
	dict_ticker['value'] = ticker
	option_list.append(dict_ticker)

# Specify time period for monitoring
init_date = dt.datetime(2017,1,1)
end_date = dt.datetime(2018,12,31)

## Initiate Dash app
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.layout = html.Div([

    # First set up the headers
    html.Div(html.H1("Stock Performance Tracker", style = dict(color = '#0A537D', fontSize = 48))),

    # Drop down menu
    html.Div([html.H4('Select a stock symbol:', style = dict(color = '#7f7f7f', fontSize = 24)),
        dcc.Dropdown(
            id = "Stock-Input",
            options= option_list,
            value = "SHOP",
            clearable=False)
    ]),

    # Stock time series
    html.Div([
        dcc.Graph(
            id = "Price-Over-Time"
        )
    ]),

    # Portfolio composition table
    html.Div([
        html.Div([
            dcc.Graph(
                id = "Portfolio-Composition",
                figure = fig_2
            )
        ],className = "six columns"),
    # Portfolio comparison to SP500
        html.Div([
            dcc.Graph(
                id = "Portfolio-SP500-Comparison",
                figure = fig_3
            )
        ],className = "six columns")

    ], className = "row")

])

@app.callback(dash.dependencies.Output("Price-Over-Time","figure"),
              [dash.dependencies.Input("Stock-Input","value")])

def update_fig1(input_value):
    ## Write the whole first graph into here
    # Grab data from relevant dates
    table = pdr.get_data_yahoo(input_value, init_date, end_date).reset_index()
    table = table[['Date', 'Adj Close']]
    table.rename(columns={'Adj Close': 'Stock Value'}, inplace=True)

    # Trace of selected stock
    trace_stock = go.Scatter(y = table['Stock Value'], x = table['Date'],
                             name = input_value,
                             line = dict(color='#2E93CE', width=2))
    # Combine traces into a list
    line_time = [trace_stock]

    # Specify layout
    layout_line_time = dict(title = 'Stock Price 2017 - 2019',
                            title_x = 0.5,
                            titlefont = dict(
                                family = 'sans-serif',
                                size = 28,
                                color = '#0A537D')
                            )

    # Generate figure
    fig_1 = dict(data=line_time, layout=layout_line_time)

    return fig_1

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)