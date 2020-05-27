from quantopian.algorithm import order_optimal_portfolio
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, Returns, CustomFactor
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.optimize as opt
from quantopian.pipeline.data.morningstar import Fundamentals
from quantopian.pipeline.experimental import risk_loading_pipeline


class MomentumQ(CustomFactor):
    # will give us the returns from last quarter
    inputs = [Returns(window_length=63)]
    window_length = 63

    def compute(self, today, assets, out, lag_returns):
        out[:] = lag_returns[0]


def initialize(context):
    # Schedule our rebalance function to run at the start of
    # each week, when the market opens.
    schedule_function(
        my_rebalance,
        date_rules.week_start(),
        time_rules.market_open()
    )

    # Create our pipeline and attach it to our algorithm.
    # Constraint parameters
    context.max_leverage = 1.0
    context.max_pos_size = 0.015
    context.max_turnover = 0.95

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    attach_pipeline(risk_loading_pipeline(), 'risk_loading_pipeline')


def make_pipeline():
    """
    Create our pipeline.
    """

    # Base universe set to the QTradableStocksUS.
    base_universe = QTradableStocksUS()

    momentum_quarter = MomentumQ()
    basic_eps = Fundamentals.basic_eps_earnings_reports.latest

    momentumQ_and_eps = momentum_quarter.zscore() + basic_eps.zscore()

    shorts = momentumQ_and_eps.percentile_between(0, 20, mask=base_universe)
    longs = momentumQ_and_eps.percentile_between(80, 100, mask=base_universe)

    # Filter for all securities that we want to trade.
    securities_to_trade = (shorts | longs)

    return Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'momentumQ_and_eps': momentumQ_and_eps
        },
        screen=(securities_to_trade))


def before_trading_start(context):
    """
    Get pipeline results.
    """

    # Gets our pipeline output every day.
    context.pipe_results = pipeline_output('my_pipeline')
    context.risk_loading_pipeline = pipeline_output('risk_loading_pipeline')


def my_rebalance(context, data):
    """
    Rebalance weekly.
    """

    # Calculate target weights to rebalance
    # target_weights = compute_target_weights(context, data)

    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -context.max_pos_size,
        context.max_pos_size
    )

    # Constrain target portfolio's leverage
    max_leverage = opt.MaxGrossExposure(context.max_leverage)

    # Ensure long and short books
    # are roughly the same size
    dollar_neutral = opt.DollarNeutral()

    # Constrain portfolio turnover
    max_turnover = opt.MaxTurnover(context.max_turnover)

    constrain_sector_style_risk = opt.experimental.RiskModelExposure(
        risk_model_loadings=context.risk_loading_pipeline, version=0)

    objective = opt.MaximizeAlpha(context.pipe_results.momentumQ_and_eps)

    # If we have target weights, rebalance our portfolio

    order_optimal_portfolio(
        objective=objective,
        constraints=[constrain_pos_size,
                     max_leverage,
                     dollar_neutral,
                     max_turnover,
                     constrain_sector_style_risk],
    )