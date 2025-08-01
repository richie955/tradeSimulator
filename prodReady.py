import streamlit as st
import json
import os
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

DATA_DIR = "./data/minute/Weekly"

st.title("Backtest Model")

try:
    available_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".json")])
    if not available_files:
        st.error(f"No data files found in the directory: {DATA_DIR}. Please check the path.")
        st.stop()
except FileNotFoundError:
    st.error(f"Data directory not found: {DATA_DIR}. Please make sure the directory exists.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Strike File", available_files)

with open(os.path.join(DATA_DIR, selected_file), "r") as f:
    data = json.load(f)

candles = data["candles"]
timestamps = [c[0] for c in candles]
opens = [c[1] for c in candles]
highs = [c[2] for c in candles]
lows = [c[3] for c in candles]
prices = [c[4] for c in candles]

def fix_timezone(ts):
    return ts[:-5] + ts[-5:-2] + ":" + ts[-2:]

def compute_indicators(prices):
    close_series = pd.Series(prices)
    rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    macd_indicator = ta.trend.MACD(close_series)
    macd = macd_indicator.macd()
    signal = macd_indicator.macd_signal()
    return rsi.fillna(0).tolist(), macd.fillna(0).tolist(), signal.fillna(0).tolist()

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    if len(returns) == 0 or returns.std() == 0:
        return 0
    excess_returns = returns.mean() - (risk_free_rate / 252)
    return (excess_returns / returns.std()) * np.sqrt(252)

def calculate_max_drawdown_detailed(equity_curve):
    if len(equity_curve) == 0:
        return 0, 0, 0, 0
    
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    max_drawdown = drawdown.min()
    
    max_dd_idx = drawdown.idxmin()
    peak_before_dd = peak.loc[:max_dd_idx].idxmax()
    
    dd_duration = max_dd_idx - peak_before_dd if max_dd_idx != peak_before_dd else 0
    
    peak_value = peak.loc[peak_before_dd] if peak_before_dd in peak.index else 0
    max_dd_pct = (max_drawdown / peak_value * 100) if peak_value != 0 else 0
    
    return max_drawdown, max_dd_pct, dd_duration, peak_before_dd

def close_position_eod(position, current_price, current_time, current_index, slippage_points, commission_per_trade):
    """Close position at end of day"""
    exit_price = current_price - slippage_points
    exit_reason = "end_of_day"
    
    total_qty = position["lot_size"] * position["qty_per_lot"]
    point_pnl = exit_price - position["entry_price"]
    gross_pnl = point_pnl * total_qty
    net_pnl = gross_pnl - (position["commission"] * 2)
    
    return {
        "entry_index": position["entry_index"],
        "exit_index": current_index,
        "entry_time": position["entry_time"],
        "exit_time": current_time,
        "entry_price": position["entry_price"],
        "exit_price": round(exit_price, 2),
        "lot_size": position["lot_size"],
        "qty_per_lot": position["qty_per_lot"],
        "total_qty": total_qty,
        "point_pnl": round(point_pnl, 2),
        "gross_pnl": round(gross_pnl, 2),
        "commission": round(position["commission"] * 2, 2),
        "net_pnl": round(net_pnl, 2),
        "exit_reason": exit_reason
    }

rsi_values, macd_values, signal_values = compute_indicators(prices)
times = [datetime.fromisoformat(fix_timezone(ts)) for ts in timestamps]

st.sidebar.header("Strategy Parameters")
st.sidebar.header("Strategy Selection")
strategy = st.sidebar.selectbox("Entry Strategy", ["Random", "MACD Crossover", "RSI Oversold"])

initial_sl = st.sidebar.slider("Initial Stop Loss (points)", 1, 20, 3, 1)
target_rr = st.sidebar.slider("Risk Reward Ratio", 1.0, 10.0, 3.0, 0.5)
entry_prob = st.sidebar.slider("Random Entry Probability", 0.001, 0.5, 0.01, 0.001)
trail_percent = st.sidebar.slider("Trail after % of target reached", 0.0, 1.0, 0.5, 0.1)
slippage_points = st.sidebar.number_input(
    "Slippage (points)", min_value=0.0, max_value=5.0, value=0.2, step=0.1, help="Assumed adverse slippage on each entry and exit."
)

st.sidebar.header("Position & Costs")
lot_size = st.sidebar.number_input("Lot Size", min_value=1, value=1)
qty_per_lot = st.sidebar.number_input("Quantity per Lot", min_value=1, value=100)
commission_per_trade = st.sidebar.number_input("Commission per Trade", min_value=0.0, value=1.0, step=0.5, help="Fixed commission cost per trade (applied on entry and exit).")

st.sidebar.header("Advanced Settings")
breakeven_percent = st.sidebar.slider("Move to breakeven at % of target", 0.0, 1.0, 0.7, 0.1)
risk_free_rate = st.sidebar.number_input("Risk-free Rate (annual %)", min_value=0.0, max_value=10.0, value=2.0, step=0.1, help="Annual risk-free rate for Sharpe ratio calculation") / 100
close_eod = st.sidebar.checkbox("Close positions at end of day", value=True, help="Automatically close any open position at the end of each trading day")

if st.sidebar.button("Run Backtest"):
    np.random.seed(42)
    trades = []
    position = None
    current_day = None

    for i in range(1, len(prices)):
        price = prices[i]
        low = lows[i]
        high = highs[i]
        open_price = opens[i]
        current_time = times[i]
        
        # Check if we've moved to a new day
        if current_day is None:
            current_day = current_time.date()
        
        day_changed = current_time.date() != current_day
        
        # Close position at end of day if enabled and day has changed
        if close_eod and day_changed and position is not None:
            trade = close_position_eod(position, price, current_time, i, slippage_points, commission_per_trade)
            trades.append(trade)
            position = None
        
        # Update current day
        current_day = current_time.date()

        if position is not None:
            if low <= position["trail_sl"]:
                exit_price = position["trail_sl"] - slippage_points
                exit_reason = "stop_loss"

                total_qty = position["lot_size"] * position["qty_per_lot"]
                point_pnl = exit_price - position["entry_price"]
                gross_pnl = point_pnl * total_qty
                net_pnl = gross_pnl - (position["commission"] * 2)

                trades.append({
                    "entry_index": position["entry_index"],
                    "exit_index": i,
                    "entry_time": position["entry_time"],
                    "exit_time": times[i],
                    "entry_price": position["entry_price"],
                    "exit_price": round(exit_price, 2),
                    "lot_size": position["lot_size"],
                    "qty_per_lot": position["qty_per_lot"],
                    "total_qty": total_qty,
                    "point_pnl": round(point_pnl, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "commission": round(position["commission"] * 2, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": exit_reason
                })
                position = None

            elif high >= position["take_profit"]:
                exit_price = position["take_profit"]
                exit_reason = "take_profit"

                total_qty = position["lot_size"] * position["qty_per_lot"]
                point_pnl = (exit_price - position["entry_price"]) - slippage_points
                gross_pnl = point_pnl * total_qty
                net_pnl = gross_pnl - (position["commission"] * 2)

                trades.append({
                    "entry_index": position["entry_index"],
                    "exit_index": i,
                    "entry_time": position["entry_time"],
                    "exit_time": times[i],
                    "entry_price": position["entry_price"],
                    "exit_price": round(exit_price, 2),
                    "lot_size": position["lot_size"],
                    "qty_per_lot": position["qty_per_lot"],
                    "total_qty": total_qty,
                    "point_pnl": round(point_pnl, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "commission": round(position["commission"] * 2, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": exit_reason
                })
                position = None

            elif high > position["highest_price"]:
                position["highest_price"] = high
                progress_to_target = (high - position["entry_price"]) / (position["take_profit"] - position["entry_price"])

                if progress_to_target >= trail_percent:
                    new_trail_sl = position["highest_price"] - initial_sl
                    position["trail_sl"] = max(position["trail_sl"], new_trail_sl)

                if high >= position["breakeven_level"]:
                    position["trail_sl"] = max(position["trail_sl"], position["entry_price"])

        if position is None:
            should_enter = False
            if strategy == "Random":
                should_enter = np.random.rand() < entry_prob
            elif strategy == "MACD Crossover" and i > 1:
                if macd_values[i-2] < signal_values[i-2] and macd_values[i-1] > signal_values[i-1]:
                    should_enter = True
            elif strategy == "RSI Oversold":
                if rsi_values[i-1] < 30:
                    should_enter = True

            if should_enter:
                entry_price_base = open_price
                entry_price_final = entry_price_base + slippage_points

                sl_level = entry_price_base - initial_sl
                tp_level = entry_price_base + (initial_sl * target_rr)

                position = {
                    "entry_index": i,
                    "entry_price": entry_price_final,
                    "entry_time": times[i],
                    "initial_sl": sl_level,
                    "take_profit": tp_level,
                    "trail_sl": sl_level,
                    "highest_price": entry_price_base,
                    "lot_size": lot_size,
                    "qty_per_lot": qty_per_lot,
                    "breakeven_level": entry_price_base + (initial_sl * target_rr * breakeven_percent),
                    "commission": commission_per_trade
                }

    # Close any remaining position at the end of data
    if position is not None:
        exit_price = prices[-1] - slippage_points
        total_qty = position["lot_size"] * position["qty_per_lot"]
        point_pnl = exit_price - position["entry_price"]
        gross_pnl = point_pnl * total_qty
        net_pnl = gross_pnl - (position["commission"] * 2)

        trades.append({
            "entry_index": position["entry_index"],
            "exit_index": len(prices) - 1,
            "entry_time": position["entry_time"],
            "exit_time": times[-1],
            "entry_price": position["entry_price"],
            "exit_price": round(exit_price, 2),
            "lot_size": position["lot_size"],
            "qty_per_lot": position["qty_per_lot"],
            "total_qty": total_qty,
            "point_pnl": round(point_pnl, 2),
            "gross_pnl": round(gross_pnl, 2),
            "commission": round(position["commission"] * 2, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": "end_of_data"
        })

    st.subheader("Trade Summary")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["duration_minutes"] = trades_df.apply(lambda row: int((row["exit_time"] - row["entry_time"]).total_seconds() / 60), axis=1)

        st.dataframe(trades_df[["entry_time", "exit_time", "duration_minutes", "entry_price", "exit_price", "lot_size", "total_qty", "point_pnl", "gross_pnl", "commission", "net_pnl", "exit_reason"]])

        win_trades = trades_df[trades_df["net_pnl"] > 0]
        loss_trades = trades_df[trades_df["net_pnl"] <= 0]
        equity = trades_df["net_pnl"].cumsum()
        peak = equity.cummax()
        drawdown = equity - peak
        
        max_drawdown, max_dd_pct, dd_duration, peak_before_dd = calculate_max_drawdown_detailed(equity)
        
        if len(trades_df) > 1:
            daily_returns = trades_df.groupby(trades_df["entry_time"].dt.date)["net_pnl"].sum()
            returns_series = pd.Series(daily_returns.values)
            sharpe_ratio = calculate_sharpe_ratio(returns_series, risk_free_rate)
        else:
            sharpe_ratio = 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades_df))
            st.metric("Win Rate", f"{(len(win_trades) / len(trades_df)):.1%}" if len(trades_df) > 0 else "N/A")
            st.metric("Total Net P&L", f"{trades_df['net_pnl'].sum():,.2f}")
        with col2:
            avg_win = win_trades["net_pnl"].mean() if not win_trades.empty else 0
            avg_loss = loss_trades["net_pnl"].mean() if not loss_trades.empty else 0
            st.metric("Avg Win", f"{avg_win:,.2f}")
            st.metric("Avg Loss", f"{avg_loss:,.2f}")
            st.metric("Avg Net P&L / Trade", f"{trades_df['net_pnl'].mean():,.2f}")
        with col3:
            profit_factor = win_trades["net_pnl"].sum() / abs(loss_trades["net_pnl"].sum()) if not loss_trades.empty and loss_trades["net_pnl"].sum() != 0 else np.inf
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            st.metric("Payoff Ratio", f"{(avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf:.2f}")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Risk-adjusted return measure")
        with col4:
            st.metric("Max Drawdown ($)", f"{max_drawdown:,.2f}")
            st.metric("Max Drawdown (%)", f"{max_dd_pct:.1f}%")
            st.metric("DD Duration (trades)", f"{dd_duration:.0f}", help="Number of trades in max drawdown period")

        st.subheader("Interactive Price Chart with Trade Entries and Exits")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times, 
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='white', width=1),
            hovertemplate='<b>Price</b>: %{y:.2f}<br><b>Time</b>: %{x}<extra></extra>'
        ))
        
        if not win_trades.empty:
            fig.add_trace(go.Scatter(
                x=win_trades["entry_time"], 
                y=win_trades["entry_price"],
                mode='markers',
                name='Win Entry',
                marker=dict(color='green', symbol='triangle-up', size=10),
                hovertemplate='<b>Win Entry</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=win_trades["exit_time"], 
                y=win_trades["exit_price"],
                mode='markers',
                name='Win Exit',
                marker=dict(color='blue', symbol='triangle-down', size=10),
                hovertemplate='<b>Win Exit</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
        
        if not loss_trades.empty:
            fig.add_trace(go.Scatter(
                x=loss_trades["entry_time"], 
                y=loss_trades["entry_price"],
                mode='markers',
                name='Loss Entry',
                marker=dict(color='red', symbol='triangle-up', size=10),
                hovertemplate='<b>Loss Entry</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=loss_trades["exit_time"], 
                y=loss_trades["exit_price"],
                mode='markers',
                name='Loss Exit',
                marker=dict(color='purple', symbol='triangle-down', size=10),
                hovertemplate='<b>Loss Exit</b><br>Price: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Trade Entry & Exit Points",
            xaxis_title="Time",
            yaxis_title="Price",
            hovermode='closest',
            showlegend=True,
            height=600,
            xaxis=dict(rangeslider=dict(visible=True)),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("PnL Distribution")
        fig_hist = px.histogram(
            trades_df, 
            x="net_pnl", 
            nbins=25, 
            title="Net PnL Distribution",
            labels={'net_pnl': 'Net PnL', 'count': 'Number of Trades'}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Equity Curve and Drawdown")
        fig_equity = make_subplots(rows=2, cols=1, 
                                   shared_xaxes=True,
                                   subplot_titles=('Cumulative P&L', 'Drawdown'),
                                   vertical_spacing=0.1)
        
        fig_equity.add_trace(
            go.Scatter(x=equity.index, y=equity, name="Cumulative P&L", line=dict(color='blue')),
            row=1, col=1
        )
        
        fig_equity.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", 
                      fill='tonexty', fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')),
            row=2, col=1
        )
        
        fig_equity.update_layout(height=600, title_text="Equity Curve and Drawdown Analysis")
        fig_equity.update_xaxes(title_text="Trade Number", row=2, col=1)
        fig_equity.update_yaxes(title_text="Cumulative P&L", row=1, col=1)
        fig_equity.update_yaxes(title_text="Drawdown", row=2, col=1)
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Win/Loss Breakdown")
            if len(win_trades) > 0 or len(loss_trades) > 0:
                fig_pie = px.pie(
                    values=[len(win_trades), len(loss_trades)], 
                    names=["Wins", "Losses"],
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_b:
            st.subheader("Exit Reason Breakdown")
            exit_counts = trades_df["exit_reason"].value_counts()
            fig_bar = px.bar(
                x=exit_counts.index, 
                y=exit_counts.values,
                labels={'x': 'Exit Reason', 'y': 'Count'},
                title="Count of Exit Reasons"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.subheader("Daily Trade Statistics")
        trades_df["trade_date"] = trades_df["entry_time"].dt.date
        daily_stats = trades_df.groupby("trade_date").agg(
            total_net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            trade_count=("net_pnl", "count"),
            avg_duration_min=("duration_minutes", "mean"),
            total_commission=("commission", "sum")
        )
        st.dataframe(daily_stats.style.format("{:,.2f}"))
        
        st.subheader("Commission Summary")
        total_commission = trades_df["commission"].sum()
        total_gross_pnl = trades_df["gross_pnl"].sum()
        commission_impact = (total_commission / total_gross_pnl) * 100 if total_gross_pnl > 0 else 0
        
        comm_col1, comm_col2, comm_col3 = st.columns(3)
        comm_col1.metric("Total Commission Paid", f"{total_commission:,.2f}")
        comm_col2.metric("Gross P&L (pre-commission)", f"{total_gross_pnl:,.2f}")
        comm_col3.metric("Commission Impact", f"{commission_impact:.2f}%", help="Percentage of gross profit consumed by commissions.")

    else:
        st.warning("No trades were executed with the current settings. Try increasing the entry probability or adjusting other parameters.")