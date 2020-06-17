from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output

# Import specific filters and factors which will be used
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import BusinessDaysSincePreviousEvent

# Import datasets which will be used
from quantopian.pipeline.data.factset.estimates import PeriodicConsensus, Actuals
from quantopian.pipeline.data.sentdex import sentiment

# import optimize
import quantopian.optimize as opt

# Import pandas
import pandas as pd


def initialize(context):
    """
    Initialize constants, create pipeline, and schedule functions
    """

    # Constants for min and max estimate surprise, sentiment, and days from announcement
    context.MIN_SHORT_SURPRISE = .025
    context.MAX_LONG_SURPRISE = -.025

    context.MIN_SHORT_SENTIMENT = 0.0
    context.MAX_LONG_SENTIMENT = 0.0

    context.MAX_DAYS_AFTER_EARNINGS_TO_OPEN = 3
    context.MAX_DAYS_AFTER_EARNINGS_TO_HOLD = 40

    # Constants for the min and max weights. for shorts these are negative
    context.MIN_WEIGHT = -.05
    context.MAX_WEIGHT = .05

    # Make our pipeline and attach to the algo
    attach_pipeline(make_pipeline(context), 'earnings_pipe')

    # Place orders
    schedule_function(
        func=place_orders,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_open(hours=2, minutes=29)
    )


def make_pipeline(context):
    """
    Define our pipeline.
    This implements the logic determining longs and shorts
    """

    # Create datasets of sales estimates and actuals for the most recent quarter (fq0).
    fq0_eps_cons = PeriodicConsensus.slice('EPS', 'qf', 0)
    fq0_eps_act = Actuals.slice('EPS', 'qf', 0)

    # Define factors of the last mean consensus EPS estimate and actual EPS.
    fq0_eps_cons_mean = fq0_eps_cons.mean.latest
    fq0_eps_act_value = fq0_eps_act.actual_value.latest

    # Define a surprise factor as the relative difference between the actual EPS and the final 
    # mean estimate made prior to the report being published. A positive value means the company
    # beat analyst expectations. A negative value means the company missed expectations.
    surprise = (fq0_eps_act_value - fq0_eps_cons_mean) / fq0_eps_cons_mean

    # Calculate the days since an earnings announcement impacted trading
    # The asof_date considers whether an announcement was before or after hours
    # We want to open on recent announcements but then close after an announcement is old
    days_since_announcement = BusinessDaysSincePreviousEvent(inputs=[fq0_eps_act.asof_date])
    recent_announcement = days_since_announcement <= context.MAX_DAYS_AFTER_EARNINGS_TO_OPEN
    old_announcement = days_since_announcement > context.MAX_DAYS_AFTER_EARNINGS_TO_HOLD

    # Define a sentiment factor.
    news_sentiment = sentiment.sentiment_signal.latest

    # Select short stocks which recently beat earnings and have a positive news sentiment. 
    # Likewise long stocks which recently missed earnings estimates and have a negative sentiment.
    shorts = (
            recent_announcement
            & (surprise > context.MIN_SHORT_SURPRISE)
            & (news_sentiment > context.MIN_SHORT_SENTIMENT)
    )
    longs = (
            recent_announcement
            & (surprise < context.MAX_LONG_SURPRISE)
            & (news_sentiment < context.MAX_LONG_SENTIMENT)
    )

    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'surprise': surprise,
            'old_announcement': old_announcement,
        },
        screen=QTradableStocksUS(),
    )

    return pipe


def before_trading_start(context, data):
    """
    Run our pipeline to fetch the actual data. 
    """

    context.output = pipeline_output('earnings_pipe')


def place_orders(context, data):
    """
    Use Optimize to place orders all at once
    """

    # get a list of the currently held securities.
    current_positions = list(context.portfolio.positions)

    # Make a series of the longs and shorts and associated alphas
    long_alphas = -context.output.query('index not in @current_positions and longs').surprise
    short_alphas = -context.output.query('index not in @current_positions and shorts').surprise

    # Hold any current positions which the announcement isn't old (again invert the surprise)
    hold_these_alphas = -context.output.query('index in @current_positions and not old_announcement').surprise

    # Combine the three
    all_alphas = pd.concat([long_alphas, short_alphas, hold_these_alphas])

    # Create our maximize alpha objective
    alpha_objective = opt.MaximizeAlpha(all_alphas)

    # Set the constraints
    max_gross_exposure = opt.MaxGrossExposure(1.0)
    max_position_size = opt.PositionConcentration.with_equal_bounds(
        context.MIN_WEIGHT,
        context.MAX_WEIGHT
    )
    dollar_neutral = opt.DollarNeutral()

    # Execute the order_optimal_portfolio method with above objective and constraint
    order_optimal_portfolio(
        objective=alpha_objective,
        constraints=[
            max_gross_exposure,
            max_position_size,
            dollar_neutral,
        ]
    )