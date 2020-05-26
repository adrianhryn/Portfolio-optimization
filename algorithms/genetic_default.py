import numpy as np

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


def EvaluationFunction(individual, df_assets_data, max_weight=1.0):
    constrain = False
    normW = list(np.array(individual['Genome']) / sum(individual['Genome']))
    if sum(normW) == 1.0:
        for W in normW:
            if W > max_weight:
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

            mu = ((1 + mu) ** 252) - 1
            sigma = np.sqrt(252) * sigma
            sharpe = mu / sigma
            fitness = float(sharpe)
    else:
        fitness = 0.0
        mu = 0.0

    individual['Fitness'] = fitness
    individual['ER'] = mu * 100.0
    return individual


def Evolve(HP_data, pop_size=200, generations=50):
    pop = creatPopulation(pop_size, len(HP_data.columns))
    hof = {'Geneome': [], 'Fitness': np.nan}
    for _ in range(generations):
        for i in range(len(pop)):
            pop[i] = EvaluationFunction(pop[i], HP_data)

        fitnesses = []
        for i in range(len(pop)):
            fitnesses.append(pop[i]['Fitness'])

        bestInGeneration = pop[np.argmax(fitnesses)]
        if bestInGeneration['Fitness'] > hof['Fitness'] or np.isnan(hof['Fitness']):
            hof = bestInGeneration
        # print 'Generation: '+str(_)+' Best Fit: '+str(bestInGeneration['Fitness'])
        selected = []
        for i in range(pop_size):
            selected.append(tournamentSelection(pop, 5))

        np.random.shuffle(selected)
        selected_A = selected[:int(len(selected) / 2)]
        selected_B = selected[int(len(selected) / 2):]
        nextGeneration = []
        for i in range(len(selected_A)):
            child_1, child_2 = crossover(selected_A[i], selected_B[i], 0.75)
            nextGeneration.append(child_1)
            nextGeneration.append(child_2)

        for i in range(len(nextGeneration)):
            nextGeneration[i] = mutation(nextGeneration[i], 0.25)

        pop = nextGeneration

    normGenome = list((np.array(hof['Genome']) / sum(hof['Genome'])))
    result = dict(list(zip(HP_data.columns, normGenome)))

    return hof, pop, result


'''
GA operators END
'''


def initialize(context):
    context.assets = [symbol('GLD'), symbol('LQD'),
                      symbol('AGG'), symbol('SPY')]

    context.cash_percent = 0.05
    context.port_weights = {}
    context.last_year = 0
    context.look_back = 3
    # print 'Look Back Years: '+str(context.look_back)
    # set_commission(commission.PerTrade(cost=0.0025))
    # set_benchmark(symbol('SPY'))
    schedule_function(func=rebalance,
                      date_rule=date_rules.week_start(),
                      time_rule=time_rules.market_open())


def rebalance(context, data):
    # print 'Rebalance'
    for position in context.portfolio.positions:
        found = False
        for asset in context.port_weights:
            if position.symbol == asset.symbol:
                found = True
        if found == False:
            order_target_percent(position, 0.0)

    for asset in context.port_weights:
        if data.can_trade(asset):
            W = context.port_weights[asset]
            # if asset.symbol=='AGG':
            # print 'AGG: '+str(W)
            order_target_percent(asset, W)


def Optimize(context, data):
    price_history = data.history(context.assets, "price", 252 * context.look_back, "1d")
    hof, pop, result = Evolve(price_history)
    context.port_weights = result


def before_trading_start(context, data):
    current_date = get_datetime()
    if int(current_date.year) > int(context.last_year):
        print('New Year!!!')
        Optimize(context, data)
        portSum = 0.0
        print(current_date.year)
        for asset in context.port_weights:
            context.port_weights[asset] = context.port_weights[asset] * (1.0 - context.cash_percent)
            portSum = portSum + context.port_weights[asset]
            print(str(asset.symbol) + ': ' + str(context.port_weights[asset]))
        print('Cash: ' + str(context.cash_percent))
        portSum = portSum + context.cash_percent
        print('Total Port: ' + str(portSum))
        print('__________________________')
        context.last_year = int(current_date.year)


def handle_data(context, data):
    record(leverage=context.account.leverage)