from utils.strategy_supporting_functions import *
from utils.strategy_functions import *
from utils.utility_functions import *
from datetime import date, timedelta

# Main execution
strategy_name="all"

portfolio = {
  "JLHL.BO": 0.05224,
  "CAMS.BO": 0.06639,
  "LATENTVIEW.BO": 0.03931,
  "UNIPARTS.BO": 0.09343,
  "SONACOMS.BO": 0.01947,
  "TANLA.BO": 0.00371,
  "HBLPOWER.BO": 0.01176,
  "AVANTEL.BO": 0.06605,
  "BPCL.BO": 0.01673,
  "IRCON.BO": 0.02477,
  "ROUTE.BO": 0.10233,
  "CIPLA.BO": 0.06656,
  "JISLJALEQS.BO": 0.22742,
  "JPPOWER.BO": 0.04556,
  "LXCHEM.BO": 0.04105,
  "TATATECH.BO": 0.05331,
  "KAYNES.BO": 0.02086,
  "WINDLAS.BO": 0.04905
}


today = date.today()
six_months_ago = today - timedelta(days=180)

start_date = "2001-01-01"
end_date = six_months_ago - timedelta(days=1)
initial=100000

training_data=download_data(list(portfolio.keys()), start_date, end_date)

valid_tickers=list(training_data.columns)
portfolio={k:v for k,v in portfolio.items() if k in valid_tickers}

tickers_with_index=list(training_data.columns).copy()
tickers_with_index.extend(["^BSESN","^NSEI" ])

test_data_with_indices = download_data(tickers_with_index, start_date=six_months_ago, end_date=today)
test_data_returns = test_data_with_indices.pct_change().dropna()

test_data_without_indices =test_data_with_indices.drop(["^BSESN","^NSEI" ],axis=1)
final_returns =test_data_returns.drop(["^BSESN","^NSEI" ],axis=1)
test_data_benchmark_returns=test_data_returns["^NSEI"]

if strategy_name!="all":
    weights=selected_strategy(strategy_name,training_data)

if strategy_name=="all":
    strategy_name,weights=get_best_weights(training_data)

optimized_portfolio={k:v for k,v in zip(list(valid_tickers),list(weights))}
weights_dict={strategy_name:np.array(list(optimized_portfolio.values())),"Portfolio":np.array(list(portfolio.values()))}

# print("Total Profit",calculate_profit(test_data_without_indices,weights,initial))
# print("Total Returns",total_return(final_returns.dot(list(weights)).values))
# print("Annualized Returns",annualized_return(final_returns.dot(list(weights)).values,trading_days=252))
# print("Active Returns",active_return(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values))
# print("Tracking Error",tracking_error(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values))
# print("Portfolio Volatility",portfolio_volatility(final_returns.dot(list(weights)).values))
# print("Sharpe Ratio",sharpe_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252))
# print("Sortino Ratio",sortino_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252))
# print("Portfolio Beta",portfolio_beta(final_returns.dot(list(weights)),test_data_benchmark_returns))
# print("Portfolio Alpha",portfolio_alpha(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252))
# print("Treynor Ratio",treynor_ratio(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252))
# print("Max Drawdown",maximum_drawdown(final_returns.dot(list(weights))))



fig=plot_cumulative_returns(test_data_returns,weights_dict,initial=100000,day="1D")
fig.show()