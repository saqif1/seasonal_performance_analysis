import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Seasonal Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def analyze_monthly_seasonality(ticker, target_month, hypothesis_type="outperformance", alpha=0.05, 
                               min_years=10, economic_threshold=0.005):
    """
    Perform a comprehensive quantitative analysis of monthly seasonality for a given stock.
    """
    # Validate inputs
    if hypothesis_type not in ["outperformance", "underperformance"]:
        raise ValueError("hypothesis_type must be 'outperformance' or 'underperformance'")
    
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    
    # Download historical data with progress indicator
    progress_bar = st.progress(0, text="Downloading stock data...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", auto_adjust=True)
    progress_bar.progress(30, text="Processing data...")
    
    if hist.empty:
        st.error(f"No data found for ticker {ticker}")
        return None
    
    # Calculate monthly returns
    monthly_prices = hist['Close'].resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    # Check if we have enough data
    if len(monthly_returns) < min_years * 12:
        st.error(f"Insufficient data: Only {len(monthly_returns)//12} years available, need at least {min_years}")
        return None
    
    # Create DataFrame with year and month
    returns_df = pd.DataFrame({
        'return': monthly_returns,
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'month_name': monthly_returns.index.month_name(),
        'date': monthly_returns.index
    })
    
    progress_bar.progress(60, text="Running statistical analysis...")
    
    # 1. Full sample analysis
    target_returns = returns_df[returns_df['month'] == target_month]['return']
    other_returns = returns_df[returns_df['month'] != target_month]['return']
    
    # 2. Time-split analysis (first half vs second half)
    split_year = returns_df['year'].median()
    first_half = returns_df[returns_df['year'] <= split_year]
    second_half = returns_df[returns_df['year'] > split_year]
    
    target_first = first_half[first_half['month'] == target_month]['return']
    other_first = first_half[first_half['month'] != target_month]['return']
    
    target_second = second_half[second_half['month'] == target_month]['return']
    other_second = second_half[second_half['month'] != target_month]['return']
    
    # 3. Rolling 5-year analysis
    years = sorted(returns_df['year'].unique())
    rolling_results = []
    
    for i in range(len(years) - 4):  # 5-year rolling window
        window_years = years[i:i+5]
        window_data = returns_df[returns_df['year'].isin(window_years)]
        
        target_window = window_data[window_data['month'] == target_month]['return']
        other_window = window_data[window_data['month'] != target_month]['return']
        
        if len(target_window) >= 4:  # At least 4 observations in target month
            mean_diff = target_window.mean() - other_window.mean()
            _, p_value = stats.mannwhitneyu(target_window, other_window, 
                                          alternative='greater' if hypothesis_type == "outperformance" else 'less')
            rolling_results.append({
                'start_year': window_years[0],
                'end_year': window_years[-1],
                'mean_diff': mean_diff,
                'p_value': p_value,
                'significant': p_value < alpha
            })
    
    rolling_df = pd.DataFrame(rolling_results)
    
    # 4. Control for market factors (simple version: compare to SPY)
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="max", auto_adjust=True)
        spy_monthly = spy_hist['Close'].resample('M').last().pct_change().dropna()
        
        # Align dates
        aligned_data = returns_df.set_index('date').join(spy_monthly.rename('spy_return'), how='inner')
        aligned_data['excess_return'] = aligned_data['return'] - aligned_data['spy_return']
        
        target_excess = aligned_data[aligned_data['month'] == target_month]['excess_return']
        other_excess = aligned_data[aligned_data['month'] != target_month]['excess_return']
        
        # Test if excess returns show seasonality
        _, excess_p_value = stats.mannwhitneyu(target_excess, other_excess, 
                                             alternative='greater' if hypothesis_type == "outperformance" else 'less')
    except:
        aligned_data = returns_df.set_index('date')
        aligned_data['excess_return'] = aligned_data['return']  # Fallback to raw returns
        target_excess = pd.Series(dtype=float)
        other_excess = pd.Series(dtype=float)
        excess_p_value = 1.0
    
    # Perform statistical tests on full sample
    # Normality tests
    _, normal_target_p = stats.shapiro(target_returns)
    _, normal_other_p = stats.shapiro(other_returns)
    
    # Set alternative hypothesis based on hypothesis_type
    one_sample_alternative = 'greater' if hypothesis_type == "outperformance" else 'less'
    two_sample_alternative = 'greater' if hypothesis_type == "outperformance" else 'less'
    
    # One-sample test
    if normal_target_p > alpha:
        t_stat_one, p_value_one = stats.ttest_1samp(target_returns, 0, alternative=one_sample_alternative)
        test_used_one = "One-sample t-test"
    else:
        if hypothesis_type == "outperformance":
            _, p_value_one = stats.wilcoxon(target_returns, alternative='greater')
        else:
            _, p_value_one = stats.wilcoxon(target_returns, alternative='less')
        test_used_one = "One-sample Wilcoxon test"
    
    # Two-sample test
    if normal_target_p > alpha and normal_other_p > alpha:
        t_stat_two, p_value_two = stats.ttest_ind(target_returns, other_returns, 
                                                 equal_var=False, alternative=two_sample_alternative)
        test_used_two = "Welch's t-test"
    else:
        if hypothesis_type == "outperformance":
            _, p_value_two = stats.mannwhitneyu(target_returns, other_returns, alternative='greater')
        else:
            _, p_value_two = stats.mannwhitneyu(target_returns, other_returns, alternative='less')
        test_used_two = "Mann-Whitney U test"
    
    # Calculate descriptive statistics
    target_stats = {
        'mean': target_returns.mean(),
        'std': target_returns.std(),
        'count': len(target_returns),
        'positive_percent': (target_returns > 0).sum() / len(target_returns) * 100
    }
    
    other_stats = {
        'mean': other_returns.mean(),
        'std': other_returns.std(),
        'count': len(other_returns),
        'positive_percent': (other_returns > 0).sum() / len(other_returns) * 100
    }
    
    # Performance difference
    performance_diff = target_stats['mean'] - other_stats['mean']
    
    # Check economic significance
    economically_significant = abs(performance_diff) > economic_threshold
    
    # Create visualizations
    # 1. Seasonal bar chart
    monthly_avg = returns_df.groupby(['month', 'month_name'])['return'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('month')
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['return'],
        marker_color=['red' if m == target_month else 'lightblue' for m in monthly_avg['month']]
    ))
    fig1.update_layout(
        title=f"Average Monthly Returns for {ticker}",
        xaxis_title="Month",
        yaxis_title="Average Return",
        yaxis_tickformat=".2%",
        showlegend=False,
        height=400
    )
    
    # 2. Fixed Heatmap - Only show available years
    heatmap_data = returns_df.pivot_table(index='year', columns='month', values='return', aggfunc='mean')
    
    # Sort by year in descending order and filter out any empty years
    heatmap_data = heatmap_data.sort_index(ascending=False)
    heatmap_data = heatmap_data.dropna(how='all')  # Remove years with no data
    
    # Create a more readable heatmap with proper sizing
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Prepare text for each cell
    text_data = heatmap_data.applymap(lambda x: f"{x:.1%}" if not pd.isna(x) and abs(x) >= 0.01 else 
                                    (f"{x:.2%}" if not pd.isna(x) else ""))
    
    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=month_names,
        y=heatmap_data.index.astype(str),  # Use actual available years
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title="Return", tickformat=".1%"),
        hoverongaps=False,
        hoverinfo='x+y+z',
        hovertemplate='<b>Year</b>: %{y}<br><b>Month</b>: %{x}<br><b>Return</b>: %{z:.2%}<extra></extra>',
        text=text_data.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig2.update_layout(
        title=f"Monthly Returns Heatmap for {ticker}",
        xaxis_title="Month",
        yaxis_title="Year",
        height=600,
        autosize=True
    )
    
    # 3. Seasonal decomposition
    returns_series = returns_df.set_index('date')['return']
    returns_series = returns_series.asfreq('M')
    
    decomposition = STL(returns_series, period=12).fit()
    
    fig3 = make_subplots(rows=4, cols=1, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    
    fig3.add_trace(go.Scatter(x=returns_series.index, y=returns_series, mode='lines', name='Observed'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=returns_series.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig3.add_trace(go.Scatter(x=returns_series.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig3.add_trace(go.Scatter(x=returns_series.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)
    
    fig3.update_layout(height=600, title_text="Seasonal Decomposition of Returns", showlegend=False)
    
    # 4. Target month distribution
    fig4 = go.Figure()
    fig4.add_trace(go.Box(y=target_returns, name=f"Month {target_month}", boxpoints='all', jitter=0.3))
    fig4.add_trace(go.Box(y=other_returns, name="Other months", boxpoints='all', jitter=0.3))
    fig4.update_layout(
        title=f"Return Distribution: Month {target_month} vs Other Months",
        yaxis_tickformat=".2%",
        yaxis_title="Return",
        height=400
    )
    
    # 5. Rolling analysis plot
    if not rolling_df.empty:
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=rolling_df['end_year'], y=rolling_df['mean_diff'], 
                                 mode='lines+markers', name='5-Year Rolling Difference'))
        fig5.add_hline(y=0, line_dash="dash", line_color="gray")
        fig5.update_layout(
            title=f"5-Year Rolling Performance Difference: Month {target_month} vs Other Months",
            xaxis_title="End Year of 5-Year Window",
            yaxis_title="Performance Difference",
            yaxis_tickformat=".2%",
            height=400
        )
    
    # 6. Time-split analysis plot
    split_labels = [f"{first_half['year'].min()}-{first_half['year'].max()}", 
                   f"{second_half['year'].min()}-{second_half['year'].max()}"]
    split_target_means = [target_first.mean(), target_second.mean()]
    split_other_means = [other_first.mean(), other_second.mean()]
    
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=split_labels, y=split_target_means, name=f'Month {target_month}'))
    fig6.add_trace(go.Bar(x=split_labels, y=split_other_means, name='Other Months'))
    fig6.update_layout(
        title=f"Time-Split Analysis: Month {target_month} vs Other Months",
        xaxis_title="Time Period",
        yaxis_title="Average Return",
        yaxis_tickformat=".2%",
        barmode='group',
        height=400
    )
    
    progress_bar.progress(100, text="Analysis complete!")
    progress_bar.empty()
    
    # Prepare results for display
    month_name = monthly_avg[monthly_avg['month'] == target_month]['month_name'].iloc[0]
    
    return {
        'ticker': ticker,
        'target_month': target_month,
        'month_name': month_name,
        'hypothesis_type': hypothesis_type,
        'alpha': alpha,
        'economic_threshold': economic_threshold,
        'target_returns': target_returns,
        'other_returns': other_returns,
        'target_stats': target_stats,
        'other_stats': other_stats,
        'p_value_one_sample': p_value_one,
        'p_value_two_sample': p_value_two,
        'test_used_one': test_used_one,
        'test_used_two': test_used_two,
        'heatmap_data': heatmap_data,
        'performance_difference': performance_diff,
        'significant': p_value_two < alpha,
        'economically_significant': economically_significant,
        'time_split_analysis': {
            'first_half': {'target': target_first.mean(), 'other': other_first.mean()},
            'second_half': {'target': target_second.mean(), 'other': other_second.mean()}
        },
        'rolling_analysis': rolling_df,
        'excess_returns': {
            'target': target_excess.mean() if len(target_excess) > 0 else None,
            'other': other_excess.mean() if len(other_excess) > 0 else None,
            'p_value': excess_p_value
        },
        'normal_target_p': normal_target_p,
        'normal_other_p': normal_other_p,
        'figures': {
            'monthly_avg': fig1,
            'heatmap': fig2,
            'decomposition': fig3,
            'distribution': fig4,
            'rolling': fig5 if not rolling_df.empty else None,
            'time_split': fig6
        },
        'years_range': f"{returns_df['year'].min()} - {returns_df['year'].max()}",
        'target_count': len(target_returns),
        'other_count': len(other_returns)
    }

def display_results(result):
    """Display the analysis results in a clean Streamlit layout"""
    
    # Header
    st.title(f"Seasonal Analysis of {result['ticker']}")
    
    # Key metrics row
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=f"{result['month_name']} Avg Return", 
            value=f"{result['target_stats']['mean']:.2%}",
            delta=f"{result['performance_difference']:+.2%} vs other months"
        )
    
    with col2:
        sig_icon = "✅" if result['significant'] else "❌"
        st.metric(
            label="Statistical Significance", 
            value=f"{sig_icon} p={result['p_value_two_sample']:.4f}"
        )
    
    with col3:
        econ_icon = "✅" if result['economically_significant'] else "❌"
        st.metric(
            label="Economic Significance", 
            value=f"{econ_icon} {abs(result['performance_difference']):.2%}"
        )
    
    with col4:
        st.metric(
            label="Positive Months", 
            value=f"{result['target_stats']['positive_percent']:.1f}%",
            delta=f"{result['target_stats']['positive_percent'] - result['other_stats']['positive_percent']:+.1f}% vs other months"
        )
    
    # Charts section
    st.subheader("Visual Analysis")
    
    # First row of charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(result['figures']['monthly_avg'], use_container_width=True)
    with col2:
        st.plotly_chart(result['figures']['distribution'], use_container_width=True)
    
    # Heatmap
    st.plotly_chart(result['figures']['heatmap'], use_container_width=True)
    
    # Third row of charts
    col1, col2 = st.columns(2)
    with col1:
        if result['figures']['rolling']:
            st.plotly_chart(result['figures']['rolling'], use_container_width=True)
    with col2:
        st.plotly_chart(result['figures']['time_split'], use_container_width=True)
    
    # Seasonal decomposition
    st.plotly_chart(result['figures']['decomposition'], use_container_width=True)
    
    # Detailed statistics section
    st.subheader("Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"{result['month_name']} Performance")
        st.write(f"**Mean Return:** {result['target_stats']['mean']:.4%}")
        st.write(f"**Standard Deviation:** {result['target_stats']['std']:.4%}")
        st.write(f"**Positive Months:** {result['target_stats']['positive_percent']:.1f}%")
        st.write(f"**Number of Observations:** {result['target_count']}")
    
    with col2:
        st.info("Other Months Performance")
        st.write(f"**Mean Return:** {result['other_stats']['mean']:.4%}")
        st.write(f"**Standard Deviation:** {result['other_stats']['std']:.4%}")
        st.write(f"**Positive Months:** {result['other_stats']['positive_percent']:.1f}%")
        st.write(f"**Number of Observations:** {result['other_count']}")
    
    # Performance comparison
    st.info("Performance Comparison")
    st.write(f"**Performance Difference:** {result['performance_difference']:+.4%}")
    st.write(f"**Economic Significance Threshold:** {result['economic_threshold']:.3%}")
    st.write(f"**Economically Significant:** {'✅ YES' if result['economically_significant'] else '❌ NO'}")
    
    # Statistical tests
    st.subheader("Statistical Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Normality Tests")
        normal_target = "Normal" if result['normal_target_p'] > result['alpha'] else "Not Normal"
        normal_other = "Normal" if result['normal_other_p'] > result['alpha'] else "Not Normal"
        st.write(f"**Target Month:** p-value = {result['normal_target_p']:.4f} ({normal_target})")
        st.write(f"**Other Months:** p-value = {result['normal_other_p']:.4f} ({normal_other})")
    
    with col2:
        st.info("Hypothesis Tests")
        st.write(f"**One-sample test:** {result['test_used_one']}")
        st.write(f"p-value: {result['p_value_one_sample']:.4f}")
        st.write(f"**Two-sample test:** {result['test_used_two']}")
        st.write(f"p-value: {result['p_value_two_sample']:.4f}")
        st.write(f"**Significance Level:** α = {result['alpha']}")
    
    # Time-split analysis
    st.info("Time-Split Analysis")
    first_half = result['time_split_analysis']['first_half']
    second_half = result['time_split_analysis']['second_half']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**First Half of Data:**")
        st.write(f"{result['month_name']}: {first_half['target']:.4%}")
        st.write(f"Other months: {first_half['other']:.4%}")
        st.write(f"Difference: {first_half['target'] - first_half['other']:+.4%}")
    
    with col2:
        st.write("**Second Half of Data:**")
        st.write(f"{result['month_name']}: {second_half['target']:.4%}")
        st.write(f"Other months: {second_half['other']:.4%}")
        st.write(f"Difference: {second_half['target'] - second_half['other']:+.4%}")
    
    # Market-adjusted analysis
    if result['excess_returns']['target'] is not None:
        st.info("Market-Adjusted Analysis (vs SPY)")
        st.write(f"**{result['month_name']} Excess Return:** {result['excess_returns']['target']:.4%}")
        st.write(f"**Other Months Excess Return:** {result['excess_returns']['other']:.4%}")
        st.write(f"**Difference:** {result['excess_returns']['target'] - result['excess_returns']['other']:+.4%}")
        st.write(f"**p-value:** {result['excess_returns']['p_value']:.4f}")
    
    # Conclusion
    st.subheader("Conclusion")
    
    if result['p_value_two_sample'] < result['alpha'] and result['economically_significant']:
        # Check if effect is consistent across time periods
        first_half_diff = result['time_split_analysis']['first_half']['target'] - result['time_split_analysis']['first_half']['other']
        second_half_diff = result['time_split_analysis']['second_half']['target'] - result['time_split_analysis']['second_half']['other']
        
        if result['hypothesis_type'] == "outperformance":
            consistent_effect = (first_half_diff > 0) and (second_half_diff > 0)
        else:
            consistent_effect = (first_half_diff < 0) and (second_half_diff < 0)
        
        if consistent_effect:
            st.success(f"✅ STRONG evidence supports {result['month_name']} {result['hypothesis_type']} hypothesis")
            st.write(f"{result['month_name']} has consistently {'outperformed' if result['hypothesis_type'] == 'outperformance' else 'underperformed'} other months by {abs(result['performance_difference']):.4%} on average")
        else:
            st.warning(f"✅ MODERATE evidence supports {result['month_name']} {result['hypothesis_type']} hypothesis")
            st.write(f"{result['month_name']} has {'outperformed' if result['hypothesis_type'] == 'outperformance' else 'underperformed'} other months by {abs(result['performance_difference']):.4%} on average")
            st.write("BUT: Effect is not consistent across all time periods")
    elif result['p_value_two_sample'] < result['alpha']:
        st.info(f"✅ STATISTICAL but not ECONOMIC evidence for {result['month_name']} {result['hypothesis_type']}")
        st.write("The effect is statistically significant but too small to be economically meaningful")
    else:
        st.error(f"❌ No reliable evidence for {result['month_name']} {result['hypothesis_type']}")
        st.write("Any observed pattern is likely due to random chance")

# Main app
def main():
    # Sidebar for user inputs
    with st.sidebar:
        st.title("Analysis Parameters")
        
        # Ticker input
        ticker = st.text_input("Stock Ticker", value="SPY").upper()
        
        # Month selection
        month_names = ["January", "February", "March", "April", "May", "June", 
                      "July", "August", "September", "October", "November", "December"]
        target_month = st.selectbox("Target Month", options=list(range(1, 13)), 
                                  format_func=lambda x: month_names[x-1])
        
        # Hypothesis type
        hypothesis_type = st.radio("Hypothesis Type", 
                                 ["outperformance", "underperformance"])
        
        # Statistical parameters
        alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
        economic_threshold_percent = st.slider(
            "Economic Significance Threshold (%)", 
            0.1, 2.0, 0.5, 0.1, 
            format="%.1f%%"
        )
        economic_threshold = economic_threshold_percent / 100
        min_years = st.slider("Minimum Years of Data", 5, 20, 10)
        
        # Run analysis button
        run_analysis = st.button("Run Analysis", type="primary")
    
    # Run analysis when button is clicked
    if run_analysis:
        with st.spinner("Running analysis..."):
            result = analyze_monthly_seasonality(
                ticker, target_month, hypothesis_type, alpha, min_years, economic_threshold
            )
            
        if result:
            display_results(result)
        else:
            st.error("Analysis failed. Please check your inputs and try again.")

if __name__ == "__main__":
    main()