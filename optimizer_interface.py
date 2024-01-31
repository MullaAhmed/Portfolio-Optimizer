import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px

from utils.strategy_supporting_functions import *
from utils.strategy_functions import *
from utils.utility_functions import *

strategies={ 
    "Auto Select Strategy": "all",
    "Equal Weights Strategy": "equal_weights",
    "Maximum Diversification Strategy": "max_div_weights",
    "Equal Risk Contribution Strategy": "erc_weights",
    "Minimum Absolute Deviation Strategy": "mad_weights",
    "Risk Parity Strategy": "risk_parity_weights",
    "Black-Litterman Model Strategy": "bl_weights",
    "Global Minimum Variance Strategy": "gmv_weights",
    "Conditional Value at Risk Strategy": "cvar_weights",
    "Momentum-Based Strategy": "momentum_weights",
    "Bayesian Optimization Strategy": "bayesian_optimization_weights",
    "Robust Optimization Strategy": "robust_optimization_weights",
    "Genetic Algorithm Optimization Strategy": "genetic_algorithm_optimization_weights",
    "Stochastic Optimization Strategy": "stochastic_optimization_weights",
    "Variational Quantum Eigensolver (VQE) Strategy": "vqe",
    "Quantum Approximate Optimization Algorithm (QAOA) Strategy": "qaoa"}

# Streamlit app layout
st.title('Portfolio Analysis Tool')

# User input for strategy name
strategy_name_input = st.selectbox('Select Strategy', list(strategies.keys()))
strategy_name=strategies[strategy_name_input]

# Portfolio input - either table input or file upload
st.subheader('Enter Your Portfolio')
portfolio_input_method = st.radio("Choose the portfolio input method", ('Table', 'Upload CSV'))

if portfolio_input_method == 'Table':
    # Create an editable table for portfolio input
    default_data = pd.DataFrame([
        {"Ticker": "TCS.BO", "Invested Amount": 10000.0},
        # Add other default rows
    ])
    
    tickers = list(pd.read_csv("all_stocks.csv")["Tickers"].values)

    default_data["Ticker"] = (
        default_data["Ticker"].astype("category").cat.add_categories(tickers)
    )

    edited_df = st.data_editor(default_data,num_rows="dynamic",width=1000)
    old_portfolio = {k:v for k,v in zip(edited_df["Ticker"].values,edited_df["Invested Amount"].values)}
    


elif portfolio_input_method == 'Upload CSV':
    # File uploader widget
    old_portfolio={"TCS.BO":10000.0}
    portfolio_input_file = st.radio("Choose the platform", ('Angel One', '5Paisa'))

    if portfolio_input_file=="Angel One":

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            old_portfolio = read_angel_one_portfolio(uploaded_file)
    
    elif portfolio_input_file=="5Paisa":
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            old_portfolio = read_5paisa_portfolio(uploaded_file)
       



# Date range for data download

today = date.today()
six_months_ago = today - timedelta(days=180)
start_date = "2001-01-01"
end_date = six_months_ago - timedelta(days=1)

# Downloading data
if st.button('Optimize Portfolio'): 
    if len(old_portfolio.keys())>1:
        progress_bar = st.progress(0)

        old_portfolio,initial=portfolio_to_weights(old_portfolio)
        # Convert DataFrame to a dictionary format for the portfolio
        
        training_data = download_data(list(old_portfolio.keys()), start_date, end_date)
        if training_data.shape[0]!=0:

            valid_tickers=list(training_data.columns)
            invalid_tickers=" ".join([i for i in list(old_portfolio.keys()) if i not in valid_tickers])
            
            if len(invalid_tickers)>0:
                st.warning("These tickers are invalid becase they were recently listed:")
                st.warning(str(invalid_tickers))
            
            old_portfolio={k:v for k,v in old_portfolio.items() if k in valid_tickers}
            old_portfolio=dict(sorted(old_portfolio.items()))

            tickers_with_index=list(valid_tickers).copy()
            tickers_with_index.extend(["^BSESN","^NSEI" ])

            progress_bar.progress(20)
            
            test_data_with_indices = download_data(tickers_with_index, start_date=six_months_ago, end_date=today)
            test_data_returns = test_data_with_indices.pct_change().dropna()

            progress_bar.progress(40)

            test_data_without_indices =test_data_with_indices.drop(["^BSESN","^NSEI" ],axis=1)
            final_returns =test_data_returns.drop(["^BSESN","^NSEI" ],axis=1)
            test_data_benchmark_returns=test_data_returns["^NSEI"]

            if strategy_name!="all":
                weights=selected_strategy(strategy_name,training_data)

            if strategy_name=="all":
                strategy_name,weights=get_best_weights(training_data)

            progress_bar.progress(80)

            optimized_portfolio={k:v for k,v in zip(list(valid_tickers),list(weights))}
            optimized_portfolio=dict(sorted(optimized_portfolio.items()))
            
            strategy_name=get_key_from_value(strategies,strategy_name)
            
            weights_dict={strategy_name:np.array(list(optimized_portfolio.values())),"Portfolio":np.array(list(old_portfolio.values()))}
            
            old_weights=np.array(list(old_portfolio.values()))
            st.subheader("Strategy Used: "+str(strategy_name))

            fig1, fig2 = st.columns(2)
            
            st.write(old_portfolio.keys(),optimized_portfolio.keys())
            old_portfolio_df = pd.DataFrame({'Ticker': list(old_portfolio.keys()),'Weight': old_weights})        
            fig_old = px.pie(old_portfolio_df, values='Weight', names='Ticker', title='Old Portfolio Weights')
            fig_old.update_layout(margin=dict(l=20, r=20, t=20, b=20), width=350, height=350)
            fig1.plotly_chart(fig_old)

            # Create Pie Chart for New (Optimized) Portfolio
            new_portfolio_df = pd.DataFrame({'Ticker': list(optimized_portfolio.keys()),'Weight': list(optimized_portfolio.values())})
            fig_new = px.pie(new_portfolio_df, values='Weight', names='Ticker', title='New Portfolio Weights')
            fig_new.update_layout(margin=dict(l=20, r=20, t=20, b=20), width=350, height=350)
            fig2.plotly_chart(fig_new)


            col11, col12, col13 = st.columns(3)
            col11.metric("Potential Profit",calculate_profit(test_data_without_indices,weights,initial),np.round(calculate_profit(test_data_without_indices,weights,initial)-calculate_profit(test_data_without_indices,old_weights,initial),5))
            col12.metric("Total Returns",total_return(final_returns.dot(list(weights)).values),np.round(total_return(final_returns.dot(list(weights)).values)-total_return(final_returns.dot(list(old_weights)).values),5))
            col13.metric("Annualized Returns",annualized_return(final_returns.dot(list(weights)).values,trading_days=252),np.round(annualized_return(final_returns.dot(list(weights)).values,trading_days=252)-annualized_return(final_returns.dot(list(old_weights)).values,trading_days=252),5))
            
            col21, col22, col23 = st.columns(3)
            col21.metric("Active Returns",active_return(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values),np.round(active_return(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values)-active_return(final_returns.dot(list(old_weights)).values,test_data_benchmark_returns.values),5))
            col22.metric("Tracking Error",tracking_error(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values),np.round(tracking_error(final_returns.dot(list(weights)).values,test_data_benchmark_returns.values)-tracking_error(final_returns.dot(list(old_weights)).values,test_data_benchmark_returns.values),5))
            col23.metric("Portfolio Volatility",portfolio_volatility(final_returns.dot(list(weights)).values),np.round(portfolio_volatility(final_returns.dot(list(weights)).values)-portfolio_volatility(final_returns.dot(list(old_weights)).values),5))
            
            col31, col32, col33 = st.columns(3)
            col31.metric("Sharpe Ratio",sharpe_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252),np.round(sharpe_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252)-sharpe_ratio(final_returns.dot(list(old_weights)).values,0.07,trading_days=252),5))
            col32.metric("Sortino Ratio",sortino_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252),np.round(sortino_ratio(final_returns.dot(list(weights)).values,0.07,trading_days=252)-sortino_ratio(final_returns.dot(list(old_weights)).values,0.07,trading_days=252),5))
            col33.metric("Treynor Ratio",treynor_ratio(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252),np.round(treynor_ratio(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252)-treynor_ratio(final_returns.dot(list(old_weights)).values,test_data_benchmark_returns,0.07,trading_days=252),5))

            col41, col42, col43 = st.columns(3)
            col41.metric("Portfolio Alpha",portfolio_alpha(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252),np.round(portfolio_alpha(final_returns.dot(list(weights)).values,test_data_benchmark_returns,0.07,trading_days=252)-portfolio_alpha(final_returns.dot(list(old_weights)).values,test_data_benchmark_returns,0.07,trading_days=252),5))
            col42.metric("Portfolio Beta",portfolio_beta(final_returns.dot(list(weights)),test_data_benchmark_returns),np.round(portfolio_beta(final_returns.dot(list(weights)),test_data_benchmark_returns)-portfolio_beta(final_returns.dot(list(old_weights)),test_data_benchmark_returns),5))
            col43.metric("Max Drawdown",maximum_drawdown(final_returns.dot(list(weights))),np.round(maximum_drawdown(final_returns.dot(list(weights)))-maximum_drawdown(final_returns.dot(list(old_weights))),5))

            

            fig=plot_cumulative_returns(test_data_returns,weights_dict,initial=initial,day="1D")
            st.plotly_chart(fig)
            progress_bar.progress(100)
        else:
            st.error("Please re-try again.")
    else:
        st.error("Please input the portfolio data.")
