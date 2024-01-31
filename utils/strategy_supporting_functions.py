import numpy as np

def shrinkage_covariance(returns,delta):
        sample_cov = returns.cov().values
        prior = np.diag(returns.var())
        return delta * prior + (1 - delta) * sample_cov

def constraints_and_initial_guess(tickers):
    """
    Generate constraints and an initial guess for portfolio optimization.
    """
    cons = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_guess = [1. / len(tickers) for _ in tickers]
    return cons, bounds, initial_guess

def max_diversification(weights, returns):
    """
    Objective function to maximize diversification.
    """
    portfolio_covariance = np.dot(returns.cov() * 252, weights)
    portfolio_variance = np.dot(weights.T, portfolio_covariance)
    return -np.sum(weights * portfolio_covariance) / np.sqrt(portfolio_variance)

def equal_risk_contribution(weights, returns):
    """
    Objective function for equal risk contribution.
    """
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    asset_contributions = weights * np.dot(returns.cov() * 252, weights) / portfolio_volatility
    return np.sum(np.square(asset_contributions - asset_contributions.mean()))

def mad(weights, returns):
    """
    Calculate Mean Absolute Deviation of a portfolio.
    """
    portfolio_return = np.dot(returns.mean(), weights)
    deviations = returns.dot(weights) - portfolio_return
    return np.sum(np.abs(deviations))

def risk_parity(weights, returns, no_tickers):
    """
    Calculate risk parity of a portfolio.
    """
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    asset_contributions = weights * (np.dot(returns.cov() * 252, weights)) / portfolio_volatility
    return np.sum(np.square(asset_contributions - 1.0 / no_tickers))

def bl(weights, returns, bl_return):
    """
    Calculate Black-Litterman return.
    """
    return -weights.dot(bl_return) + 0.5 * weights.T.dot(returns.cov() * 252).dot(weights)

def gmv(weights, returns):
    """
    Calculate Global Minimum Variance of a portfolio.
    """
    return np.dot(weights.T, np.dot(returns.cov() * 252, weights))

def mean_cvar(weights, returns, alpha):
    """
    Calculate Mean Conditional Value at Risk.
    """
    portfolio_returns = returns.dot(weights)
    threshold = np.percentile(portfolio_returns, alpha * 100)
    return -np.mean(portfolio_returns[portfolio_returns < threshold])