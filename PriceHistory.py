import pandas as pd
import numpy as np
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import time
from requests.exceptions import ContentDecodingError

def corr_helper(row, *args):
    correl = args[0]
    try:
        return correl['SPY'][row.Date][row.ticker]
    except KeyError:
        return np.nan

def beta_helper(row, *args):
    spy = args[0]
    try:
        return (row['corr'] * row['vol']) / spy.loc[row['Date'], 'vol']
    except KeyError:
        return np.nan


class PriceHistory:

    def __init__(self, tickers: list):
        self.dfs = dict()
        self.tickers = tickers
        self.combined_price_history = None

    def get_data(self, overwrite=False):
        if not overwrite:
            try:
                self.combined_price_history = pd.read_pickle('combined_price_history.pkl')
            except FileNotFoundError:
                self._get_data()
        else:
            self._get_data()

    def _get_data(self):
        successes = 0
        tries = 0
        for t in self.tickers:
            success = False
            print(t)
            fails = 0
            tries += 1
            while fails <= 3 and not success:
                try:
                    df = data.DataReader(t.upper(), 'yahoo', '1980-01-01')
                    df['ticker'] = t.upper()
                    df['Date'] = df.index
                    df['pct_change'] = df['Close'].pct_change().shift(-1)
                    df['log_return'] = (np.log(df.Close) - np.log(df.Close.shift(1))).shift(-1)
                    df['vol'] = df['pct_change'].rolling(252).std()
                    df['month_log_return'] = pd.to_numeric(
                        pd.rolling_sum(df['log_return'][::-1], window=22, min_periods=22)[::-1])
                    df['week_log_return'] = pd.to_numeric(pd.rolling_sum(df['log_return'][::-1], window=5, min_periods=5)[::-1])
                    df['slow'] = df['Close'].rolling(25).mean()
                    df['fast'] = df['Close'].rolling(7).mean()
                    df['momentum'] = (df['fast'] - df['slow']) / df['slow']
                    df['value'] = (df['Close'] - df['Close'].rolling(252*3).mean()) / df['Close'].rolling(252*3).mean()
                    df['dollar_volume'] = df['Volume'] * df['Close']
                    df['average_dollar_volume'] = df['dollar_volume'].rolling(22).mean()
                    self.dfs[t] = df
                    successes += 1
                    success = True
                except (RemoteDataError, ContentDecodingError):
                    fails += 1
                    time.sleep(3)
            print("Succeeded to get data for {} of {} tickers".format(successes, tries))

        self.combined_price_history = pd.concat(list(self.dfs.values()))
        self.process_data()
        pd.to_pickle(self.combined_price_history, 'combined_price_history.pkl')

    def process_data(self):
        df = self.combined_price_history.dropna()
        df['Close'] = pd.to_numeric(df['Close'])
        df['Date'] = pd.to_datetime(df['Date'])

        gb_equity = pd.DataFrame()
        for key, grp in df.groupby('ticker'):
            gb_equity[key] = grp['log_return']
        correl = gb_equity.rolling(252).corr()
        df['corr'] = df.apply(corr_helper, axis=1, args=(correl,))
        df['beta'] = df.apply(beta_helper, axis=1, args=(df.loc[df.ticker == 'SPY'],))
        df['beta_abs'] = df['beta'].abs()
        df['ra_return'] = df['log_return'] / df['beta_abs']

        security_encodings = dict((s, i) for i, s in enumerate(df['ticker'].unique()))

        df['security_enc'] = df['ticker'].apply(lambda x: security_encodings[x])
        df['scaled_volume'] = np.log(pd.to_numeric(df['Volume'].values) + 1)
        df['volume_rank'] = 0
        df['vol_category'] = 0
        for key, grp in df.groupby('Date'):
            df.loc[df.Date == key, 'vol_category'] = pd.qcut(grp.vol, 10, labels=range(10))
            df.loc[df.Date == key, 'volume_rank']  = grp['average_dollar_volume'].rank() - 1

        df['volume_rank'] = df['volume_rank'].astype(int)
        df['vol_category'] = df['vol_category'].astype(int)

        self.combined_price_history = df.dropna()
        pd.to_pickle(self.combined_price_history, 'combined_price_history.pkl')

    def gen_train_test(self, feature_cols, target_col, ts=.4):
        return train_test_split(self.combined_price_history[feature_cols],
                                self.combined_price_history.loc[:, ['Date', 'log_return', target_col]],
                                test_size=ts)

    def gen_train_test_ts(self, feature_cols, target_col):
        X_train, y_train, X_test, y_test = [], [], [], []
        for key, grp in self.combined_price_history.groupby('ticker'):
            tscv = TimeSeriesSplit(n_splits=int(len(grp)/2.))
            for tr, te in tscv.split(grp):
                X_train.append(grp.iloc[tr, :].loc[:, feature_cols])
                X_test.append(grp.iloc[te, :].loc[:, feature_cols])

                y_train.append(grp.iloc[tr, :].loc[:, target_col])
                y_test.append(grp.iloc[te, :].loc[:, target_col])


        X_train = pd.concat(X_train, ignore_index=True)
        X_test = pd.concat(X_test, ignore_index=True)
        y_train = pd.concat(y_train, ignore_index=True)
        y_test = pd.concat(y_test, ignore_index=True)
        return X_train, y_train, X_test, y_test
