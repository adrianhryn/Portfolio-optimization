# Long-short approach in portfolio optimization problem

## Structure:
1. Introduction
2. Why is that important?
3. Method descriptions
4. Results and plots
5. How to reproduce our work?
6. Resources that were of great help for us


## Introduction

Traditional approaches for portfolio trading are conservative and not as effective as they can be. They aim for interpretability and concentrate mostly on Markowitz framework. We want to bring some novelty into this. Our main goal is to find out how various ML approaches perform on the task of portfolio optimization and its trading. Of course, interpretability is still the most important part.

We will update Readme.md with new approaches as long as we work out them.

## Why is that important?

There is a convention in science and statistics of investitions that most investors don’t bring too much to the table. In other words, their added value for trading strategies is small. They could be right about general trading rules and have some insights, but they use non-effective approaches in portfolio construction. We want to decrease the human factor in weights allocations metter, although it is very important to intelligently select trading factors (it is the place where the model interpretability mostly comes from). Our mutual approach should unite the meaningful human insights with ML optimizations approaches.

## Method descriptions

All the good strategies consist of meaningful factors that significantly reflect causes of changes in companies values. So this is crucial to do factor analysis before you run any algorithms. Backtest is only a tool and it shouldn't go as a final argument, feature importance in ML or factor analysis in portfolio optimization are key elements of robust models and strategies. You can see an example of such analysis in ‘analysis_notebooks/alpha_factor_analysis.ipynb’

For now on, we ran several experiments with long/short strategies using Quantopian platform. We also used Genetic Algorithm for traditional approach as optimization algorithm for weights allocations. If you want to get acquainted with all details, check our poster in 'results/presentations' directory and Google presentation [here](https://docs.google.com/presentation/d/19knJVxAn4K7khZVep67Lgszupuyzt1Je9Mu0rliJmJ0/edit?usp=sharing). There are many visualizations and more detailed explanations.

## Results and plots

First round stands for model selection, we preserve the same time period and SPY index as performance benchmark for all experiments. The second round stands for testing on out of sample data, it consists of non-overlapping with first round time period. There is the long-short approach and the traditional one. There is no finetuning of parameters in the second round, we used the same factors and the same algorithms parameters of the first round winners (see the figures with perfomance below). And, of course, we did factor analysis and ran the backtest only with meaningful factors, which we know from alphalens framework.

**Table 1: First and second rounds of experiments**
![Results](https://github.com/adrianhryn/Portfolio-optimization/blob/master/results/results_table.png)

You can see below how constructed portfolios in the second round perform on the out of sample periods.

It is worth adding that third long-short algorithm passed all the constraints to the Quantopian contest, you can find them [here](https://www.quantopian.com/contest). It also gave better average results comparing to the market index.

**Figure 1: Quantopian optimizer + eps and momentum as combined factor**
![Figure 1](https://github.com/adrianhryn/Portfolio-optimization/blob/master/results/out_of_sample_plots/long_short_quantopian_api.png)

The portfolio based on Genetic algorithm optimizer could be a better option for people that want to buy the market index, because it performed noticeably better on the last two year period. Although it fell to much during the start of the Covid crisis, but then it went up with great speed comparing to the market index.

**Figure 2: Genetic algorithm optimizer + eps and momentum as combined factor**
![Figure 2](https://github.com/adrianhryn/Portfolio-optimization/blob/master/results/out_of_sample_plots/traditional_genetic.png)

If you want to get acquainted with all details, check our poster in 'results/presentations' directory and presentation [here](https://docs.google.com/presentation/d/19knJVxAn4K7khZVep67Lgszupuyzt1Je9Mu0rliJmJ0/edit?usp=sharing). There are many visualizations and more detailed explanations.

## How to reproduce our work?

In our work, we used Quantopian API which you can not run outside of Quantopian environment.
So, in order to run our code you should sign up and log in [here](https://www.quantopian.com/posts)
Then you can upload our notebook from 'analysis_notebooks' directory and run it from top to down.

If you want to see the algorithms performance and their statistics, you should:
1. Click 'New Algorithm'
2. Upload a .py python file
3. Choose the backtest period
4. Click 'Build Algorithm'
5. Click 'Run Full Backtest'
6. Get the statistics

All the requirements to libraries versions are satisfied inside the Quantopian platform.
All the data you need you can query through the code using Quantopian API (which is implemented in our code).
Each of the .py files in the 'algorithms/' directory is a distinct algorithm which you can easily run if you follow the six steps above.

## Resources that were of great help for us
- Quantopian platform. Their lectures, tutorials, examples and community forum are beautiful and very usefull for comperhensive research.

Next steps to concetrate:
- Medium blog of Alex Honchar about Neural Networks for timeseries predictions
- [Deep Learning for Predicting Asset Returns](https://arxiv.org/abs/1804.09314) by Guanhao Feng, Jingyu He, Nicholas G. Polson
