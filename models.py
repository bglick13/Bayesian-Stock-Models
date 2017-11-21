import pymc3 as pm
from PriceHistory import PriceHistory
import numpy as np
from theano import shared
import matplotlib.pyplot as plt
import seaborn as sns
from tickers import tickers
import pandas as pd


def invlogit(x):
    return np.exp(x) / (1 + np.exp(x))


class Model:
    def __init__(self, factor, metric='week_log_return', vol=False):
        self.factor = factor
        self.metric = metric
        self.vol = vol
        self.ph = PriceHistory(tickers)
        self.ph.get_data(overwrite=False)

        feature_cols = [factor, 'scaled_volume', 'vol', 'security_enc', 'volume_rank', 'ticker', 'vol_category']
        self.X_train, self.X_test, self.y_train, self.y_test = self.ph.gen_train_test(feature_cols, metric)
        self.output = None

    def train(self, n_samples=100):
            with self.model:
                self.trace = pm.sample(n_samples, init='advi', n_init=50000, step=pm.NUTS(), tune=200)
            _df = pm.trace_to_dataframe(self.trace)
            pd.to_pickle(_df, '{}_trace.pkl'.format(self.name))

    def predict(self):
        self.y_test['prediction'] = self.X_test.progress_apply(self._predict_helper, axis=1)
        self.output = self.y_test

    def eval(self):
        fig, ax = plt.subplots(figsize=(17, 8))
        sns.jointplot(self.output['prediction'], self.output[self.metric], ax=ax)

    def backtest(self):
        fig, ax = plt.subplots(figsize=(17, 8))
        self.output['prediction'] = 0
        self.output.loc[self.output['prediction'] >= .55, 'position'] = 1
        self.output.loc[self.output['prediction'] <= .45, 'position'] = -1
        self.output['bt_return'] = self.output['log_return'] * self.output['position']
        self.output.sort_values('Date').groupby('Date').sum()['bt_return'].cumsum().plot(ax=ax)
        self.output.sort_values('Date').groupby('Date').sum()['log_return'].cumsum().plot(ax=ax)


class VolumeCategorized(Model):
    def __init__(self, factor, metric='week_log_return', vol=False):
        super().__init__(factor, metric, vol)
        self.name = 'volume'
        cutoff = self.ph.combined_price_history.volume_rank.max() - 100

        _X_train = self.X_train[self.X_train.volume_rank >= cutoff]
        self.y_train = self.y_train[self.X_train.volume_rank >= cutoff]
        _X_test = self.X_test[self.X_test.volume_rank >= cutoff]
        self.y_test = self.y_test[self.X_test.volume_rank >= cutoff]
        self.X_train = _X_train
        self.X_test = _X_test

        self.X_train['volume_rank'] = (self.X_train['volume_rank'] - cutoff).astype(int)
        self.X_test['volume_rank'] = (self.X_test['volume_rank'] - cutoff).astype(int)

        self.n_volumes = int(self.ph.combined_price_history.volume_rank.max() - 100)
        self.n_categories = int(10)

        with pm.Model() as self.model:
            # Hyperpriors for slope based on vol classification
            self.a_category_mu = pm.StudentT('a_category_mu', mu=0, sd=3, nu=3)
            self.a_category_sd = pm.HalfCauchy('a_category_sd', 2)

            # Hyperpriors for slope, derived from vol_classification, used for individual security calculatio
            self.a_sec = pm.Normal('a_security', mu=self.a_category_mu, sd=self.a_category_sd, shape=self.n_volumes)
            if self.vol:
                self.a_vol = pm.Normal('a_vol', mu=0, sd=3)

            # Calculate the intercept for each security independently, nothing fancy here
            self.b_sec = pm.Uniform('b_security', -1, 1, shape=self.n_categories)
            if self.vol:
                p = invlogit((self.X_train[self.factor].values * self.a_sec[self.X_train['volume_rank'].values] +
                              self.X_train['vol'].values * self.a_vol +
                              self.b_sec[self.X_train['vol_category'].values]))
            else:
                p = invlogit((self.X_train[self.factor].values * self.a_sec[self.X_train['volume_rank'].values] +
                              self.b_sec[self.X_train['vol_category'].values]))
            self.likelihood = pm.Bernoulli('likelihood', p=p,
                                           observed=np.where(self.y_train[self.metric].values > 0, 1, 0))

    def _predict_helper(self, row):
        if self.vol:
            a = invlogit((
                row[self.factor] * self.trace['a_security'][:, row.volume_rank] +
                row.vol * self.trace['a_vol'] *
                self.trace['b_security'][:, row.vol_category]))
        else:
            a = invlogit((
                row[self.factor] * self.trace['a_security'][:, row.volume_rank] +
                self.trace['b_security'][:, row.vol_category]))
        return np.mean(a)

    def print_output(self):
        dists = dict(a_security=[], b_security=[])
        for a in self.trace['a_security'].transpose():
            dists['a_security'].append(np.array(a))
        for a in self.trace['b_security'].transpose():
            dists['b_security'].append(np.array(a))
        print(dists)


class EveryStock(Model):
    def __init__(self, factor, metric='week_log_return', vol=False):
        super().__init__(factor, metric, vol)
        self.name = 'every_stock'
        securities = set(self.X_test.ticker.unique()) | set(self.X_train.ticker.unique())
        n_securities = len(securities)
        sec_map = dict((t, i) for i, t in enumerate(securities))
        self.X_train['security_enc'] = self.X_train.ticker.apply(lambda x: sec_map[x])
        self.X_test['security_enc'] = self.X_test.ticker.apply(lambda x: sec_map[x])
        self.sec_category_map = (pd.concat([self.X_test, self.X_train]).groupby(['security_enc', 'ticker']).size()
                            .reset_index()['ticker'].values)

        with pm.Model() as self.model:
            # Hyperpriors for slope based on vol classification
            a_category_mu = pm.StudentT('a_category_mu', mu=0, sd=3, nu=3)
            a_category_sd = pm.HalfCauchy('a_category_sd', 2)

            # Hyperpriors for slope, derived from vol_classification, used for individual security calculatio
            a_sec = pm.Normal('a_security', mu=a_category_mu, sd=a_category_sd, shape=n_securities)
            #     a_volume = pm.Normal('a_volume', mu=0, sd=3)
            if self.vol:
                a_vol = pm.Normal('a_vol', mu=0, sd=3)

            # Calculate the intercept for each security independently, nothing fancy here
            b_sec = pm.Uniform('b_security', -1, 1, shape=n_securities)

            if self.vol:
                p = invlogit((self.X_train[self.factor].values * a_sec[self.X_train['security_enc'].values] +
                              self.X_train['vol'].values * a_vol +
                              b_sec[self.X_train['security_enc'].values]))
            else:
                p = invlogit((self.X_train[self.factor].values * a_sec[self.X_train['security_enc'].values] +
                              b_sec[self.X_train['security_enc'].values]))

            likelihood = pm.Bernoulli('likelihood', p=p,
                                      observed=np.where(self.y_train[self.metric].values > 0, 1, 0))
            #     mom_trace = pm.sample(500, init='advi', n_init=50000, step=pm.NUTS())

    def _predict_helper(self, row):
        if self.vol:
            a = invlogit((
                row[self.factor] * self.trace['a_security'][:, row.security_enc] +
                row.vol * self.trace['a_vol'] *
                self.trace['b_security'][:, row.security_enc]))
        else:
            a = invlogit((
                row[self.factor] * self.trace['a_security'][:, row.security_enc] +
                self.trace['b_security'][:, row.security_enc]))
        return np.mean(a)

    def print_output(self):
        dists = {}
        for v in self.trace.varnames:
            dists[v] = {}
            _a = self.trace.get_values(v)
            if len(_a.shape) > 1:
                for i, d in enumerate(_a.transpose()):
                    dists[v][self.sec_category_map[i]] = d
            else:
                dists[v]['sec'] = _a
        print(dists)


class NonHierarchical(Model):
    def __init__(self, factor, metric='week_log_return', vol=False):
        super().__init__(factor, metric, vol)
        self.name = 'non_hierarchical'

        with pm.Model() as self.model:
            a_sec = pm.Normal('a_security', mu=0, sd=3)

            # Calculate the intercept for each security independently, nothing fancy here
            b_sec = pm.Uniform('b_security', -1, 1, shape=10)

            p = invlogit((self.X_train[self.factor].values * a_sec +
                          b_sec[self.X_train['vol_category'].values]))

            likelihood = pm.Bernoulli('likelihood', p=p,
                                      observed=np.where(self.y_train[self.metric].values > 0, 1, 0))

    def _predict_helper(self, row):
        a = invlogit((
            row[self.factor] * self.trace['a_security'] +
            self.trace['b_security'][:, row.vol_category]))
        return np.mean(a)

    def print_output(self):
        dists = {}
        for v in self.trace.varnames:
            dists[v] = {}
            _a = self.trace.get_values(v)
            if len(_a.shape) > 1:
                for i, d in enumerate(_a.transpose()):
                    dists[v][self.sec_category_map[i]] = d
            else:
                dists[v]['sec'] = _a
        print(dists)