import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import patsy
import os, sys

def vol_category_helper(row, *args):
    q0, q1, q2, q3 = args
    if row <= q0:
        return 0
    elif row <= q1:
        return 1
    elif row <= q2:
        return 2
    elif row <= q3:
        return 3
    else:
        return 3


def beta_helper(row, *args):
    spy = args[0]
    try:
        return pd.Series(dict(beta=(row['corr'] * row['vol']) / spy.loc[row['Date'], 'vol'],
                              market_log_return=spy.loc[row['Date'], 'log_return']))
    except KeyError:
        return pd.Series(dict(beta=np.nan, market_log_return=np.nan))


def corr_helper(row, *args):
    correl = args[0]
    try:
        return correl['SPY'][row.Date][row.ticker]
    except KeyError:
        return np.nan


def calc_momentum(combined, fast_window=7, slow_window=25):
    for ticker in combined['ticker'].unique():
        combined.loc[combined['ticker'] == ticker, 'fast'] = \
            combined.loc[combined['ticker'] == ticker, 'Close'].rolling(fast_window).mean()

        combined.loc[combined['ticker'] == ticker, 'slow'] = \
            combined.loc[combined['ticker'] == ticker, 'Close'].rolling(slow_window).mean()

    return pd.to_numeric((combined['fast'] - combined['slow']) / combined['slow'])


def calc_value(combined, slow_window=252*3):
    for ticker in combined['ticker'].unique():
        combined.loc[combined['ticker'] == ticker, 'slow'] = \
            combined.loc[combined['ticker'] == ticker, 'Close'].rolling(slow_window).mean()

    return pd.to_numeric((combined['Close'] - combined['slow']) / combined['slow'])


def build_prices_dfs():
    root_dir = os.getcwd()
    out = dict()
    tickers = os.listdir(root_dir + '/data/equities') + os.listdir(root_dir + '/data')
    for ticker in tickers:
        if '.csv' in ticker:
            try:
                df = pd.read_csv(root_dir + '/data/equities/{}'.format(ticker))
            except FileNotFoundError:
                df = pd.read_csv(root_dir + '/data/{}'.format(ticker))
            df = df.replace({'null': np.nan})
            df['Close'] = pd.to_numeric(df['Close'])
            df['ticker'] = ticker[:-4]
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date', False)
            df['pct_change'] = pd.to_numeric(df['Close'].pct_change().shift(-1))
            df['log_return'] = (np.log(df.Close) - np.log(df.Close.shift(1))).shift(-1)

            # Calculate vol on a 1 year rolling basis
            df['vol'] = df['pct_change'].rolling(252).std()
            df['month_return'] = pd.to_numeric(pd.rolling_sum(df['pct_change'][::-1], window=22, min_periods=22)[::-1])
            df['month_log_return'] = pd.to_numeric(
                pd.rolling_sum(df['log_return'][::-1], window=22, min_periods=22)[::-1])
            out[ticker] = df
    combined = pd.concat([_df for _df in out.values()])

    gb_equity = pd.DataFrame()
    for key, grp in combined.groupby('ticker'):
        gb_equity[key] = grp['log_return']
    correl = gb_equity.rolling(252).corr()
    combined['corr'] = combined.apply(corr_helper, axis=1, args=(correl,))
    combined['beta'] = 0
    combined['market_log_return'] = 0
    combined.loc[:, ['beta', 'market_log_return']] = combined.apply(beta_helper, axis=1,
                                                                    args=(combined.loc[combined.ticker == 'SPY'],))
    combined['alpha'] = combined['log_return'] - (combined['market_log_return'] * combined['beta'])
    combined['beta_abs'] = combined['beta'].abs()
    combined['ra_return'] = combined['log_return'] / combined['beta_abs']
    for key, grp in combined.groupby('ticker'):
        combined.loc[combined['ticker'] == key, 'ra_month_return'] = pd.rolling_sum(grp['ra_return'][::-1], window=22,
                                                                                    min_periods=22)[::-1]
    q0, q1, q2, q3 = combined['vol'].quantile([.2, .4, .6, .8])
    # Equities are placed into 4 vol categories based on their moment-in-time volatility relative to all historical
    # observed vols
    combined['category'] = combined['vol'].apply(vol_category_helper, args=(q0, q1, q2, q3))
    combined['momentum'] = calc_momentum(combined)
    combined['value'] = calc_value(combined)
    return out, combined.dropna()
