
import pandas as pd
import datetime as dt
import math


def initialize(context):

    context.use_adjusted = False
    context.active = [sid(22739), sid(22972), sid(22446), sid(23921), sid(23870)]
    context.cash = sid(23870)
    context.bill = sid(23870)

    context.leverage = 1.0

    # Weights:
    #   [wR, wC, wV, wS, eps, wIV]
    #   wi ~ zi = ( ri^wR * (1-ci)^wC / (vi^wV) * (ivi^wIV) )^(wS+eps)

    #   Golden Lama EAA :
    context.score_weights = (2.0, 1.0, 0.25, 1.0, 1e-6, 4.0)
    #   IntradayVolatility EAA : wi ~ zi = (1-ci) / (intraday_vol)
    # context.score_weights = (0.0, 0.0, 0.0, 1.0, 1e-6, 1.0)
    #   Golden Offensive EAA: wi ~ zi = (1-ci) * ri^2
    # context.score_weights = (2.0, 1.0, 0.0, 1.0, 1e-6, 0.0)
    #   Golden Defensive EAA: wi ~ zi = squareroot( ri * (1-ci) )
    # context.score_weights = (1.0, 1.0, 0.0, 0.5, 1e-6, 0.0)
    #   Equal Weighted Return: wi ~ zi = ri ^ eps
    # context.score_weights = (1.0, 0.0, 0.0, 0.0, 1e-6, 0.0)
    #   Equal Weighted Hedged: wi ~ zi = ( ri * (1-ci) )^eps
    # context.score_weights = (1.0, 1.0, 0.0, 0.0, 1e-6, 0.0)
    #   Scoring Function Test:
    # context.score_weights = (1.0, 1.0, 1.0, 1.0, 0.0, 1.0)

    context.assets = set(context.active + [context.cash, context.bill])
    context.alloc = pd.Series([0.0] * len(context.assets), index=context.assets)

    schedule_function(
        reallocate,
        date_rules.month_end(days_offset=0),
        time_rules.market_close(minutes=5)
    )

    schedule_function(
        rebalance,
        date_rules.month_end(days_offset=0),
        time_rules.market_close(minutes=5)
    )


    if context.use_adjusted:
        start_year = 2002
        end_year = dt.datetime.today().year + 1
        url_template = "http://real-chart.finance.yahoo.com/table.csv?s=%s&a=0&b=1&c=%d&d=0&e=1&f=%d&g=d&ignore=.csv"

        for sym in context.active:
            url = url_template % (sym.symbol, start_year, end_year)
            print("Fetching %s adjusted prices: %s" % (sym.symbol, url))

            fetch_csv(
                url,
                date_column='Date',
                date_format='%Y-%m-%d',
                symbol=sym,
                usecols=['Adj Close'],
                pre_func=fetch_pre,
                post_func=fetch_post
            )


def handle_data(context, data):
    record(leverage=context.portfolio.positions_value / context.portfolio.portfolio_value)


def rebalance(context, data):
    for s in context.alloc.index:
        if s in data:
            order_target_percent(s, context.alloc[s] * context.leverage)


def reallocate(context, data):
    h = make_history(context, data).ix[-280:]
    h_low = history(300, '1d', 'low').ix[-280:]
    h_high = history(300, '1d', 'high').ix[-280:]

    hm = h.resample('M', how='last')[context.active]
    hb = h.resample('M', how='last')[context.bill]
    ret = hm.pct_change().ix[-12:]

    N = len(context.active)

    non_cash_assets = list(context.active)
    non_cash_assets.remove(context.cash)

    print("---")


    # excess return momentum
    mom = (hm.ix[-1] / hm.ix[-2] - hb.ix[-1] / hb.ix[-2] + \
           hm.ix[-1] / hm.ix[-4] - hb.ix[-1] / hb.ix[-4] + \
           hm.ix[-1] / hm.ix[-7] - hb.ix[-1] / hb.ix[-7] + \
           hm.ix[-1] / hm.ix[-13] - hb.ix[-1] / hb.ix[-13]) / 22

    # nominal return correlation to equi-weight portfolio
    ew_index = ret.mean(axis=1)
    corr = pd.Series([0.0] * N, index=context.active)
    for s in corr.index:
        corr[s] = ret[s].corr(ew_index)

    vol = ret.std()

    ivol = (h_high[context.active].ix[-10:] - h_low[context.active].ix[-10:]).mean() / (
                h_high[context.active] - h_low[context.active]).mean()

    # Generalized Momentum
    # wi ~ zi = ( ri^wR * (1-ci)^wC / vi^wV / ivoli^wIV )^wS

    wR = context.score_weights[0]
    wC = context.score_weights[1]
    wV = context.score_weights[2]
    wS = context.score_weights[3]
    eps = context.score_weights[4]
    wIV = context.score_weights[5]

    z = ((mom ** wR) * ((1 - corr) ** wC) / (vol ** wV) / (ivol ** wIV)) ** (wS + eps)
    z[mom < 0.] = 0.0

    # Crash Protection
    num_neg = z[z <= 0].count()
    cpf = float(num_neg) / N
    print("cpf = %f" % cpf)

    # Security selection
    # TopN = Min( 1 + roundup( sqrt( N ), rounddown( N / 2 ) )
    top_n = min(math.ceil(N ** 0.5) + 1, N / 2)

    # Allocation
    top_z = z.order().index[-top_n:]
    print("top_z = %s" % [i.symbol for i in top_z])

    w_z = ((1 - cpf) * z[top_z] / z[top_z].sum()).dropna()
    w = pd.Series([0.0] * len(context.assets), index=context.assets)
    for s in w_z.index:
        w[s] = w_z[s]
    w[context.cash] += cpf
    print("Allocation:\n%s" % w)

    context.alloc = w


def make_history(context, data):
    if context.use_adjusted:
        df = pd.DataFrame(index=data[context.active[0]]['aclose_hist'].index, columns=context.active)
        for s in context.active:
            df[s] = data[s]['aclose_hist']
        return df
    else:
        return history(300, '1d', 'price')


def fetch_pre(df):
    df = df.rename(columns={'Adj Close': 'aclose'})
    df['aclose_hist'] = pd.Series([[]] * len(df.index), index=df.index)
    return df


def fetch_post(df):
    for i in range(0, len(df.index)):
        df['aclose_hist'].ix[-i - 1] = df['aclose'][-i - 300:][:300]
    return df