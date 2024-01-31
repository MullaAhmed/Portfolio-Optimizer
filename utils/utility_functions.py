import plotly.graph_objects as go
from difflib import SequenceMatcher
import yfinance as yf
import pandas as pd
import numpy as np

def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None  # or raise an exception, or return a default value


def download_data(tickers, start_date, end_date):
    """
    Download the adjusted closing prices for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    for i in list(data.columns):
        empty=(data[i].isnull().sum())
        if empty>(0.80*len(list(data[i]))):
            data.drop([i],axis=1,inplace=True)
    data.dropna( inplace=True)
    # Sorting in descending order


    return data

def normalize_weights(weights):
    """
    Normalize the given weights so that their sum equals 1.
    """
    weights=np.array(weights)
    weights[weights < 0] = 0
    normalized_w = weights / np.sum(weights)
    rounded_w = np.round(normalized_w[:-1], 5)
    last_weight = 1 - np.sum(rounded_w)
    rounded_w = np.append(rounded_w, last_weight)

    return rounded_w

def portfolio_to_weights(portfolio):
    """
    Convert the portfolio dictionary to a list of weights.
    """
    new_portfolio={}
    total_invested=sum(list(portfolio.values()))
    for i in portfolio.keys():    
        new_portfolio[i]=np.round(portfolio[i]/total_invested,5)
    
    return new_portfolio, total_invested

def search_similar_sentence(comapny, company_list):

    most_similar = None
    highest_similarity = 0

    for s in company_list:
        similarity = SequenceMatcher(None, comapny, s).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar = s

    return most_similar

def read_5paisa_portfolio(path):
    """
    Read the portfolio data from the given path.
    """
    all_stocks=pd.read_csv("all_stocks.csv")
    df=pd.read_csv(path).dropna()
    
    portfolio={}
    total_invested=sum([x*y for x,y in zip(list(map(float,df["Qty"].values)),list(map(float,df["AvgPrice"].values)))])

    for i in range(df.shape[0]):
        company_name=search_similar_sentence(df.iloc[i][0],all_stocks["Names"].values)
        script=all_stocks[all_stocks["Names"]==company_name]["Tickers"].values[0]
        invested=np.round(float(df.iloc[i][1]*df.iloc[i][2]),5)
        portfolio[script]=invested
    
    return portfolio

def read_angel_one_portfolio(path):
    """
    Read the portfolio data from the given path.
    """
    df=pd.read_excel(path).dropna()
    rows_with_invested_value = df[df.apply(lambda row: row.astype(str).str.contains('Invested Value').any(), axis=1)]
    df=df[1:]
    df.columns=rows_with_invested_value.iloc[0]

    portfolio={}
    total_invested=sum(list(map(float,df["Invested Value"].values)))

    for i in range(df.shape[0]):
        script=df.iloc[i][0]+".BO"
        invested=np.round(float(df.iloc[i][9]),5)
        portfolio[script]=invested
    
    return portfolio

def plot_cumulative_returns(returns_with_index, weights_dict,initial,day="15D"):
    n_day_returns = returns_with_index.resample(day).apply(lambda x: (x + 1).prod() - 1)

    fig = go.Figure()

    returns = n_day_returns.drop(["^BSESN", "^NSEI"], axis=1)
    for name, weights in weights_dict.items():
        n_day_return = returns.dot(weights)
        cumulative_return = (n_day_return + 1).cumprod() * initial
        fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, mode='lines', name=name))

    cumulative_return = (n_day_returns['^BSESN'] + 1).cumprod() * initial
    fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, mode='lines', name="Sensex"))

    cumulative_return = (n_day_returns['^NSEI'] + 1).cumprod() * initial
    fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return, mode='lines', name="Nifty"))

    fig.update_layout(title='Portfolio Cumulative Returns',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Returns')

    
    return(fig)

