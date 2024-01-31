import pandas as pd
from difflib import SequenceMatcher
import numpy as np

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

data=read_5paisa_portfolio("D:/Projects/Portfolio Optimization/5paisa.csv")
print(data)