
import numpy as np
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import random

from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE

from qiskit.circuit.library import TwoLocal

from .strategy_supporting_functions import *
from .utility_functions import *
from .evaluation_metrics import *

def get_best_weights(data):
    
    cons, bounds, initial_guess = constraints_and_initial_guess(list(data.columns))
    
    returns = data.pct_change().dropna()
    weights_dict={
            "equal_weights": get_equal_weights(list(data.columns)),
            "max_div_weights": get_max_div_weights(returns, initial_guess, bounds, cons),
            "erc_weights": get_erc_weights(returns, initial_guess, bounds, cons),
            "mad_weights": get_mad_weights(returns, initial_guess, bounds, cons),
            "risk_parity_weights": get_risk_parity_weights(returns, initial_guess, bounds, cons),
            "bl_weights": get_bl_weights(returns, initial_guess, bounds, cons),
            "gmv_weights": get_gmv_weights(returns, initial_guess, bounds, cons),
            "cvar_weights": get_cvar_weights(returns, initial_guess, bounds, cons),
            "momentum_weights": get_momentum_weights(data),
            "bayesian_optimization_weights": get_bayesian_optimized_weights(returns),
            "robust_optimization_weights": get_robust_optimized_weights(returns),
            "genetic_algorithm_optimization_weights": get_genetic_algorithm_optimized_weights(returns),
            "stochastic_optimization_weights": get_stochastic_optimized_weights(returns),
        }
    portfolio_returns_dict={}
  
    for i in weights_dict.keys():
        # portfolio_returns= total_return(returns.dot(weights_dict[i]))
        
        portfolio_profit=calculate_profit(data,weights_dict[i],initial=100000)
        portfolio_returns_dict[i]=portfolio_profit # portfolio_returns
    
    best_returns_strategy=max(portfolio_returns_dict, key=portfolio_returns_dict.get)
 
    return best_returns_strategy,weights_dict[best_returns_strategy]

def selected_strategy(strategy_name,data):
  
    tickers=list(data.columns)
    cons, bounds, initial_guess = constraints_and_initial_guess(tickers)
    returns = data.pct_change().dropna()


    strategy_dict={
            "equal_weights": get_equal_weights,
            "max_div_weights": get_max_div_weights,
            "erc_weights": get_erc_weights,
            "mad_weights": get_mad_weights,
            "risk_parity_weights": get_risk_parity_weights,
            "bl_weights": get_bl_weights,
            "gmv_weights": get_gmv_weights,
            "cvar_weights": get_cvar_weights,
            "momentum_weights": get_momentum_weights,
            "bayesian_optimization_weights": get_bayesian_optimized_weights,
            "robust_optimization_weights": get_robust_optimized_weights,
            "genetic_algorithm_optimization_weights": get_genetic_algorithm_optimized_weights,
            "stochastic_optimization_weights": get_stochastic_optimized_weights,
            "vqe":quantum_VQE,
            "qaoa":quantum_QAOA
        }

    strategy = strategy_dict.get(strategy_name)
    
    kwargs={
            "strategy_name":strategy_name,"tickers":tickers,"data":data,
            "returns":returns, "initial_guess":initial_guess, 
            "bounds":bounds,  "constraints":cons
        }
    return strategy(**kwargs)
  

def quantum_QAOA(returns,q = 0.5,budget_factor = 2,**kwargs):

    # Calculate mean and covariance from returns
    mu = np.array(returns.mean())
    sigma = np.array(returns.cov())

    # Set up the portfolio optimization problem
    budget = (returns.shape[1]) // budget_factor
    portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
    qp = portfolio.to_quadratic_program()

    optimizer = COBYLA()

    qaoa = QAOA(sampler=Sampler(), optimizer=optimizer, reps=3)

    eigensolver = MinimumEigenOptimizer(qaoa)
    result = eigensolver.solve(qp)

    weights=result.x/sum(list(result.x))
    return weights

def quantum_VQE(returns,q = 0.5,budget_factor=2,**kwargs):
    # Calculate mean and covariance from returns
    mu = np.array(returns.mean())
    sigma = np.array(returns.cov())

    # Set up the portfolio optimization problem
    budget = returns.shape[1]// budget_factor
    portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
    qp = portfolio.to_quadratic_program()
    optimizer = COBYLA()

    # Set up the VQE algorithm
    ansatz = TwoLocal(returns.shape[1], "ry", "cz", reps=3, entanglement="full")
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=optimizer)
    svqe = MinimumEigenOptimizer(svqe_mes)
    result = svqe.solve(qp)

    weights=(result.x)/sum(list(result.x))
    return weights

def get_max_div_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Optimize for maximum diversification.
    """
    solution = minimize(max_diversification, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    weights=solution.x
    return normalize_weights(weights)

def get_equal_weights(tickers,**kwargs):
    """
    Return equal weights for the given tickers.
    """
    weights = [1. / len(tickers) for _ in tickers]
    return normalize_weights(weights)

def get_erc_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Optimize for equal risk contribution.
    """
    solution = minimize(equal_risk_contribution, initial_guess, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    weights=solution.x
    return normalize_weights(weights)

def get_mad_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Get weights that minimize the Mean Absolute Deviation.
    """
    solution = minimize(mad, initial_guess, method='SLSQP', args=(returns,), bounds=bounds, constraints=constraints)
    return normalize_weights(solution.x)

def get_risk_parity_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Get weights that achieve risk parity.
    """
    no_tickers=returns.shape[1]
    solution = minimize(risk_parity, initial_guess, method='SLSQP', args=(returns, no_tickers), bounds=bounds, constraints=constraints)
    return normalize_weights(solution.x)

def get_bl_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Get weights using Black-Litterman model.
    """
    tau = 0.025
    no_tickers = len(returns.columns)
    market_cap_weights = np.array([1 / no_tickers for _ in range(no_tickers)])
    market_return = returns.mean().dot(market_cap_weights)
    bl_return = market_return + tau * np.dot(returns.cov(), market_cap_weights - initial_guess)

    solution = minimize(bl, initial_guess, method='SLSQP', args=(returns, bl_return), bounds=bounds, constraints=constraints)
    return normalize_weights(solution.x)

def get_gmv_weights(returns, initial_guess, bounds, constraints,**kwargs):
    """
    Get weights that minimize the Global Minimum Variance.
    """
    solution = minimize(gmv, initial_guess, method='SLSQP', args=(returns,), bounds=bounds, constraints=constraints)
    return normalize_weights(solution.x)

def get_cvar_weights(returns, initial_guess, bounds, constraints, alpha=0.05,**kwargs):
    """
    Get weights that minimize the Conditional Value at Risk.
    """
    solution = minimize(mean_cvar, initial_guess, method='SLSQP', args=(returns, alpha), bounds=bounds, constraints=constraints)
    return normalize_weights(solution.x)

def get_momentum_weights(data, lookback_period=126,**kwargs):
    """
    Calculate weights based on momentum.
    """
    momentum = data.pct_change(lookback_period).mean()
    weights = (momentum / momentum.sum()).values
    return normalize_weights(weights)

def get_bayesian_optimized_weights(returns,**kwargs):
    num_stocks = returns.shape[1]
    mean_prior = np.zeros(num_stocks)
    mean_posterior = returns.mean().values
    n = len(returns)
    mean_updated = (n * mean_posterior + mean_prior) / (n + 1)
    return mean_updated / np.sum(mean_updated)

def get_robust_optimized_weights(returns, delta=0.5,**kwargs):
   
    cov = shrinkage_covariance(returns,delta)
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(returns.columns))
    weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
    return normalize_weights(weights)

def get_stochastic_optimized_weights(returns, num_simulations=1000,**kwargs):
    means = returns.mean().values
    cov_matrix = returns.cov().values
    simulated_returns = np.random.multivariate_normal(means, cov_matrix, num_simulations)
    mean_simulated = np.mean(simulated_returns, axis=0)
    weights = mean_simulated / np.sum(mean_simulated)
    return normalize_weights(weights)

def get_genetic_algorithm_optimized_weights(returns,**kwargs):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)  # Ensuring weight is non-negative
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=returns.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual,**kwargs):
        w = np.array(individual)
        w = w / w.sum()  # Normalize weights so they sum to 1
        port_return = np.dot(returns.mean().values, w)
        port_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov().values, w)))
        return port_return - port_vol,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=None, halloffame=None, verbose=False)

    best_ind = tools.selBest(population, 1)[0]
    best_weights = np.array(best_ind)

    weights = best_weights / best_weights.sum()  # Ensure the returned weights sum to 1

    return normalize_weights(weights)

