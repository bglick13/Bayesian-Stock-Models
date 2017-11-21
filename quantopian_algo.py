


class Algo:

    def __init__(self, dists, prob_function):
        self.dists = dists
        self.prob_function = prob_function

    def write_algo(self):
        s = 'from quantopian.algorithm import attach_pipeline, pipeline_output' \
            'from quantopian.pipeline import Pipeline, CustomFilter' \
            'from quantopian.pipeline.filters import StaticAssets' \
            'from quantopian.pipeline.data.builtin import USEquityPricing' \
            'from quantopian.pipeline.factors import SimpleMovingAverage, AnnualizedVolatility, AverageDollarVolume' \
            'from quantopian.pipeline.factors import RollingLinearRegressionOfReturns' \
            'from quantopian.pipeline.filters.morningstar import Q1500US' \
            'import numpy as np' \
            'from numpy import array as array' \
            'from numpy import float32 as float32' \
            'import pandas as pd' \
            'def initialize(context):' \
            '   context.dists = {}' \
            '   # Rebalance every day, 1 hour after market open.' \
            '   schedule_function(my_rebalance, date_rules.week_start(), time_rules.market_open(hours=1))' \
            '   # Record tracking variables at the end of each day.' \
            '   schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())' \
            '   # Create our dynamic stock selector.' \
            '   context.returns_length = 2' \
            '   context.regression_length = 252' \
            '   context.spy          = sid(8554)' \
            '   context.tlt          = sid(23921)'\
            '   dollar_volume = AverageDollarVolume(window_length=22)'\
            '   log.debug(context.portfolio.positions.keys())'\
            '   context._filter = dollar_volume.top(94) | InPortfolio(holdings=tuple(context.portfolio.positions.keys()))'\
            '   my_pipe = make_pipeline(context)'\
            '   attach_pipeline(my_pipe, "my_pipeline")'\
            'class InPortfolio(CustomFilter):'\
            '   inputs = []'\
            '   window_length = 1'\
            '   params = ("holdings", )'\
\
            '    def compute(self, today, assets, out, holdings):'\
            '        out[:] = np.in1d(assets, holdings)'\
\
            'def make_pipeline(context):'\
                """
                A function to create our dynamic stock selector (pipeline). Documentation on
                pipeline can be found here: https://www.quantopian.com/help#pipeline-title
                """\
            '   mean_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=7)'\
\
            '   # 30-day close price average.'\
            '   mean_30  = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=25)'\
            '   mean_3yr = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=252*3)'\
            '   close    = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=1)'\
            '   vol      = AnnualizedVolatility()'\
            '   volume   = AverageDollarVolume(window_length=22)'\
            '   percent_difference = (mean_10 - mean_30) / mean_30'\
            '   value = (close - mean_3yr) / mean_3yr'\
            '   regression = RollingLinearRegressionOfReturns('\
            '   target=context.spy,'\
            '   returns_length=context.returns_length,'\
            '   regression_length=context.regression_length,'\
            '   mask=context._filter)'\
            '   alpha = regression.alpha'\
            '   beta = regression.beta'\
            '   return Pipeline('\
            '   columns={'\
            '   "momentum": percent_difference,'\
            '   "value" : value,'\
            '   "alpha" : alpha,'\
            '   "beta"  : beta,'\
            '   "vol"   : vol,'\
            '   "volume": volume},'\
            '   screen = context._filter)'\
    'def invlogit(x):'\
    '    return np.exp(x) / (1 + np.exp(x))'\

def calc_prediction_helper(row, *args):
    context = args[0]
    factor = args[1]
    if (factor == 'value'):
        try:
            a = invlogit(row.momentum * np.array(context.dists['a_security'][int(row.volume_rank)]) + np.array(context.dists['b_security'][int(row.vol_category)]))
        except:
            log.debug(row.volume_rank)
            log.debug(row.vol_category)
            return 0
    if (factor == 'momentum'):
        try:
            a = invlogit(row.momentum * np.array(context.dists['a_security'][int(row.volume_rank)]) + np.array(context.dists['b_security'][int(row.vol_category)]))
        except:
            log.debug(row.volume_rank)
            log.debug(row.vol_category)
            return 0
    return np.mean(a)

def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
    context.output['equity'] = context.output.index
    context.output['volume_rank'] = context.output['volume'].rank()
    context.output['vol_category'] = pd.qcut(context.output['vol'], 10, labels=range(10))
    context.output['momentum_prob'] = context.output.apply(calc_prediction_helper, axis=1, args=(context, 'momentum'))
    context.output['value_prob'] = context.output.apply(calc_prediction_helper, axis=1, args=(context, 'momentum'))

    # Go long in securities for which the 'longs' value is True.
    mom_low, mom_high = context.output['momentum_prob'].quantile([.45, .55])
    context.momentum_longs = context.output[context.output['momentum_prob'] >= mom_high].index.tolist()

    # Go short in securities for which the 'shorts' value is True.
    context.momentum_shorts = context.output[context.output['momentum_prob'] <= mom_low].index.tolist()

    val_low, val_high = context.output['value_prob'].quantile([.45, .55])
    context.value_longs = context.output[context.output['value_prob'] >= val_high].index.tolist()
    context.value_shorts = context.output[context.output['value_prob'] <= val_low].index.tolist()

    context.longs = (set(context.momentum_longs) - set(context.value_shorts)) | (set(context.value_longs) - set(context.momentum_shorts))

    context.shorts = (set(context.momentum_shorts) - set(context.value_longs)) | (set(context.value_shorts) - set(context.momentum_longs))
    # context.output.loc[context.output['prob'] >= .5, 'weight'] = context.output.loc[context.output['prob'] >= .5, 'prob']  / len(context.longs)

    # context.output.loc[context.output['prob'] < .5, 'weight'] = -1 * context.output.loc[context.output['prob'] < .5, 'prob']  / len(context.longs)

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index

def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    # Compute even target weights for our long positions and short positions.
    if len(context.longs) > 0:
        context.long_weight = .5 / len(context.longs)
    else:
        context.long_weight = 0

    if len(context.shorts) > 0:
        context.short_weight = -.5 / len(context.shorts)
    else:
        context.short_weight = 0
    # next_beta = 0
    # df = context.output.fillna(0)
    # for security in context.longs:
    #     if data.can_trade(security):
    #         next_beta += df.loc[security, 'beta']
    # for security in context.shorts:
    #     if data.can_trade(security):
    #         next_beta -= df.loc[security, 'beta']
    # scale = next_beta / .3
    # spy_target_pct = .3
    # log.debug('Predicted Beta: {}'.format(next_beta))
    # next_leverage = len(context.longs) + len(context.shorts) + spy_target_pct
    # scale *= next_leverage
    # log.debug('Prescaled Leverage: {}'.format(next_leverage))
    # context.long_weight = 1 / scale
    # context.short_weight = -1 / scale
    # context.spy_weight = -1 * np.sign(next_beta) * spy_target_pct

def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing.
    """
    my_assign_weights(context, data)


    for security in context.portfolio.positions:
        if security not in context.longs and security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)


    # for security in context.output.index:
    #     if data.can_trade(security):
    #         weight = context.output.loc[security, 'weight']
    #         order_target_percent(security, weight)
    for security in context.longs:
        if data.can_trade(security):
            if security.symbol in ['SPY', 'TLT']:
                pass
            else:
                order_target_percent(security, context.long_weight)

    for security in context.shorts:
        if data.can_trade(security):
            if security.symbol in ['SPY', 'TLT']:
                pass
            else:
                order_target_percent(security, context.short_weight)

    portfolio_beta = calc_beta(context)
    record(calc_beta = portfolio_beta)
    # position_value = 0
    # for ticker in context.portfolio.positions:
    #     position = context.portfolio.positions[ticker]
    #     position_value += np.abs(position.amount) * position.last_sale_price
    # percent_allocated =  position_value / (context.portfolio.portfolio_value)
    # log.debug("Percent allocation: {}".format(percent_allocated))
    # denom = (1-percent_allocated)
    # if denom == 0: denom = .01
    # target = (-portfolio_beta * percent_allocated)# / (1 - percent_allocated)
    # log.debug('Target for SPY: {}'.format(context.spy_weight))
    # order_target_percent(context.spy, context.spy_weight)

    my_record_vars(context, data)

def calc_beta(c):
    beta = 0
    df = c.output.fillna(0)
    for ticker in c.portfolio.positions:
        if ticker.symbol == 'SPY':
            continue
        position = c.portfolio.positions[ticker]
        try:
            beta += ((position.amount * position.last_sale_price) / np.abs(c.portfolio.portfolio_value)) * df.loc[position.sid, 'beta']
        except KeyError:
            order_target_percent(position.sid, 0)
    return beta




def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    record(leverage=context.account.leverage)

def handle_data(context,data):
    """
    Called every minute.
    """
    pass




'