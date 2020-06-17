import numpy as np
from quantopian.algorithm import order_optimal_portfolio
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage, Returns, CustomFactor
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.optimize as opt
from quantopian.pipeline.data.morningstar import Fundamentals
from quantopian.pipeline.experimental import risk_loading_pipeline

'''
Genetic Algorithm Operators
'''


def creatPopulation(pop_size, gen_num):
    pop = []
    for i in range(pop_size):
        genome = np.random.uniform(0.0, 100.0, gen_num).tolist()
        individual = {'Genome': genome, 'Fitness': np.nan, 'ER': np.nan}
        pop.append(individual)
    return pop


def mutation(individual, probability):
    if np.random.uniform(0.0, 1.0) < probability:
        for gen in range(len(individual['Genome'])):
            individual['Genome'][gen] = individual['Genome'][gen] + np.random.normal(0, 1)
            if individual['Genome'][gen] < 0.0:
                individual['Genome'][gen] = 0.0

    return individual


def crossover(individual_1, individual_2, probability):
    child_1 = {'Genome': [], 'Fitness': np.nan, 'ER': np.nan}
    child_2 = {'Genome': [], 'Fitness': np.nan, 'ER': np.nan}
    if np.random.uniform(0.0, 1.0) < probability:
        dice = np.random.randint(1, len(individual_1['Genome']))
        child_1['Genome'] = individual_1['Genome'][:dice] + individual_2['Genome'][dice:]
        child_2['Genome'] = individual_2['Genome'][:dice] + individual_1['Genome'][dice:]
    else:
        child_1 = individual_1
        child_2 = individual_2

    return child_1, child_2


def tournamentSelection(pop, contestants):
    tournament = []
    for i in range(contestants):
        dice = np.random.randint(0, len(pop))
        already_selected = False
        for selected in tournament:
            if np.array_equal(selected['Genome'], pop[dice]['Genome']):
                already_selected = True
                i = i - 1
                break
        if already_selected == False:
            tournament.append(pop[dice])

    fitnesses = []
    for i in range(len(tournament)):
        fitnesses.append(tournament[i]['Fitness'])

    bestFit = tournament[np.argmax(fitnesses)]
    return bestFit


def EvaluationFunction(individual, df_assets_data, min_weight=0.001, max_weight=1.0, period=63):
    constrain = False
    normW = list(np.array(individual['Genome']) / sum(individual['Genome']))
    if sum(normW) == 1.0:
        for W in normW:
            if W > max_weight or W < min_weight:
                fitness = 0.0
                mu = 0.0
                constrain = True
                break

        if constrain == False:
            # Evaluate portfolio
            returns = np.asmatrix(np.diff(df_assets_data.as_matrix(), axis=0) / df_assets_data.as_matrix()[:-1])
            returns = returns.T
            weights = np.asmatrix(normW).T

            P = np.asmatrix(np.mean(returns, axis=1))
            C = np.asmatrix(np.cov(returns))

            mu = float(P.T * weights)
            sigma = float(np.sqrt(weights.T * C * weights))

            mu = ((1 + mu) ** period) - 1
            sigma = np.sqrt(period) * sigma
            sharpe = mu / sigma
            fitness = float(sharpe)
    else:
        fitness = 0.0
        mu = 0.0

    individual['Fitness'] = fitness
    individual['ER'] = mu * 100.0
    return individual


def Evolve(HP_data, pop_size=200, generations=50, period=63):
    pop = creatPopulation(pop_size, len(HP_data.columns))
    hof = {'Geneome': [], 'Fitness': np.nan}

    for _ in range(generations):
        for i in range(len(pop)):
            pop[i] = EvaluationFunction(pop[i], HP_data, period)

        fitnesses = []
        for i in range(len(pop)):
            fitnesses.append(pop[i]['Fitness'])

        bestInGeneration = pop[np.argmax(fitnesses)]
        if bestInGeneration['Fitness'] > hof['Fitness'] or np.isnan(hof['Fitness']):
            hof = bestInGeneration
        # print 'Generation: '+str(_)+' Best Fit: '+str(bestInGeneration['Fitness'])
        n_of_contestants = 5
        selected = []
        for i in range(pop_size):
            selected.append(tournamentSelection(pop, n_of_contestants))

        np.random.shuffle(selected)
        selected_A = selected[:int(len(selected) / 2)]
        selected_B = selected[int(len(selected) / 2):]
        nextGeneration = []
        cross_over_prob = 0.75
        for i in range(len(selected_A)):
            child_1, child_2 = crossover(selected_A[i], selected_B[i], cross_over_prob)
            nextGeneration.append(child_1)
            nextGeneration.append(child_2)

        mutation_prob = 0.3
        for i in range(len(nextGeneration)):
            nextGeneration[i] = mutation(nextGeneration[i], mutation_prob)

        pop = nextGeneration

    normGenome = list((np.array(hof['Genome']) / sum(hof['Genome'])))
    result = dict(list(zip(HP_data.columns, normGenome)))
    return hof, pop, result


'''
GA operators END
'''


class MomentumQ(CustomFactor):
    # will give us the returns from last quarter
    inputs = [Returns(window_length=63)]
    window_length = 63

    def compute(self, today, assets, out, lag_returns):
        out[:] = lag_returns[0]


def make_pipeline():
    base_universe = QTradableStocksUS()

    momentum_quarter = MomentumQ()
    basic_eps = Fundamentals.basic_eps_earnings_reports.latest
    style = Fundamentals.style_score.latest
    momentumQ_style_and_eps = momentum_quarter.zscore() + basic_eps.zscore() + style.zscore()

    shorts = momentumQ_style_and_eps.percentile_between(0, 10, mask=base_universe)
    longs = momentumQ_style_and_eps.percentile_between(90, 100, mask=base_universe)

    # Filter for all securities that we want to trade.
    securities_to_trade = (shorts | longs)

    return Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'momentumQ_style_and_eps': momentumQ_style_and_eps,
            # 'assets_names': assets_names
        },
        screen=(securities_to_trade))


def initialize(context):
    context.max_leverage = 1.0
    context.max_pos_size = 0.015
    context.max_turnover = 0.95

    context.long_port_weights = {}

    context.days_period = 63
    context.look_back_periods = 4

    schedule_function(func=rebalance,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open())

    attach_pipeline(make_pipeline(), 'my_pipeline')


def get_constraints(context):
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

    constraints = [constrain_pos_size, max_leverage, dollar_neutral, max_turnover]
    return constraints


def compute_target_weights(context, data):
    get_genetic_weights(context, data)

    port_weights = {}

    # equal weighted short equity
    if context.shorts:
        short_weight = -0.5 / len(context.shorts)

    # Exit positions in our portfolio if they are not
    # in our longs or shorts lists.
    for security in context.portfolio.positions:
        if security not in context.long_port_weights.keys(
        ) and security not in context.shorts and data.can_trade(security):
            port_weights[security] = 0

    for security, long_weight in context.long_port_weights.items():
        port_weights[security] = long_weight  # *0.5

    # for security in context.shorts:
    #    port_weights[security] = short_weight

    print(len(port_weights))
    print("{} {} {} {}".format(len(context.long_port_weights), len(context.shorts),
                               sorted(context.long_port_weights.values())[-10:],
                               sorted(context.long_port_weights.values())[:10]))

    return port_weights


def rebalance(context, data):
    target_weights = compute_target_weights(context, data)
    current_date = get_datetime()
    print("REBALANCE: " + str(current_date))

    constraints = get_constraints(context)
    constraints = []
    if target_weights:
        order_optimal_portfolio(
            objective=opt.TargetWeights(target_weights),
            constraints=constraints)


def get_assets_symbols(series):
    assets_symbols = []
    for symbol, bool_v in zip(series.index, series.values):
        if bool_v:
            assets_symbols.append(symbol)
    return assets_symbols


def get_genetic_weights(context, data):
    # look for one quarter back
    context.pipe_results = pipeline_output('my_pipeline')

    context.assets_symbols = get_assets_symbols(context.pipe_results['longs'])

    days_to_look_back = int(context.days_period * context.look_back_periods)
    price_history = data.history(context.assets_symbols, "price", days_to_look_back, "1d")

    # 200, 50
    hof, pop, genetic_weights_result = Evolve(price_history, pop_size=100, generations=30)
    context.long_port_weights = genetic_weights_result


def before_trading_start(context, data):
    pipe_results = pipeline_output('my_pipeline')

    # define_shorts
    context.shorts = []
    for sec in pipe_results[pipe_results['shorts']].index.tolist():
        context.shorts.append(sec)