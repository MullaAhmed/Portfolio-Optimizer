import numpy as np

def calculate_profit(data_final,weight,initial):
    
    buy_prices=data_final.iloc[0,:].values
    sell_prices=data_final.iloc[-1,:].values
    quantity = list(map(int,(initial * weight) // buy_prices))
    selling_price = np.sum(quantity * sell_prices)
    profit = selling_price - initial

    return np.round(profit,5)


def total_return(returns):
    """ Calculate the total return of the portfolio. """
    total_return = (returns + 1).prod() - 1
    return np.round(total_return,5)

def annualized_return(returns, trading_days):
    """ Calculate the annualized return of the portfolio. """
    annualized_return = (1 + total_return(returns))**(trading_days/len(returns)) - 1
    return np.round(annualized_return,5)

def active_return(returns, benchmark_returns):
    """ Calculate the active return of the portfolio. """
    return np.round(returns.mean() - benchmark_returns.mean(),5)

def tracking_error(returns, benchmark_returns):
    """ Calculate the tracking error of the portfolio. """
    return np.round((returns - benchmark_returns).std(),5)


def portfolio_volatility(returns):
    """ Calculate the volatility of the portfolio. """
    volatility = returns.std()
    return np.round(volatility,5)

def sharpe_ratio(returns, risk_free_rate, trading_days):
    """ Calculate the Sharpe ratio of the portfolio. """
    sr = (annualized_return(returns, trading_days) - risk_free_rate) / (returns.std() * (trading_days ** 0.5))
    return np.round(sr,5)

def sortino_ratio(returns, risk_free_rate, trading_days):
    """ Calculate the Sortino ratio of the portfolio. """
    negative_returns = returns[returns < 0]
    dsd = (negative_returns.std() * (trading_days ** 0.5))
    s_ratio = (annualized_return(returns, trading_days) - risk_free_rate) / dsd
    return np.round(s_ratio,5)

def treynor_ratio(returns, benchmark_returns, risk_free_rate, trading_days):
    """ Calculate the Treynor ratio of the portfolio. """
    beta = portfolio_beta(returns, benchmark_returns)
    tr = (annualized_return(returns, trading_days) - risk_free_rate) / beta
    return np.round(tr,5)

def portfolio_beta(portfolio_returns, benchmark_returns):
    """ Calculate the beta of the portfolio. """
    covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
    covariance = covariance_matrix[0, 1]  # Covariance between portfolio and benchmark
    variance = covariance_matrix[1, 1]    # Variance of the benchmark
    beta = covariance / variance
    return np.round(beta,5)

def portfolio_alpha(returns, benchmark_returns, risk_free_rate, trading_days):
    """ Calculate the alpha of the portfolio. """
    portfolio_annual_return = annualized_return(returns, trading_days)
    benchmark_annual_return = annualized_return(benchmark_returns, trading_days)
    beta = portfolio_beta(returns, benchmark_returns)
    alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    return np.round(alpha,5)

def maximum_drawdown(returns):
    """ Calculate the maximum drawdown of the portfolio. """
    cumulative = (1 + returns).cumprod()
    max_return = cumulative.cummax()
    drawdown = (cumulative - max_return) / max_return
    max_drawdown = drawdown.min()
    return np.round(max_drawdown,5)
