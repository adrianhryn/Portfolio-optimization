# Portfolio-optimization

## Structure:
1. Introduction
2. Why is that important
3. Method descriptions
4. Some results
5. How to reproduce our work
6. Resources that were of great help for us


## Introduction

Traditional approaches for portfolio trading are conservative in general. They aim for interpretability and concentrate mostly on Markowitz framework. We want to bring some novelty into this. Our main goal is to find out how various ML approaches perform on the task of portfolio optimization and its trading. Of course, interpretability is still the most important part.

We will update Readme.md with new approaches as long as we work out them.

## Why is that important

There is a convention in science and statistics of investitions that most investors don’t bring too much to the table. In other words, their added value for trading strategies is small. They could be right about general trading rules and have some insights, but they use the non-effective approaches in portfolio construction. We want to decrease the human factor in weights allocations metter, although is very important to intelligently select trading factors (it is the place where model interpretability comes from). Our mutual approach should unite the meaningful human insights with optimization ML approaches.

## Method descriptions

All the good strategies consists of meaningful  factors that significantly reflects causes to changes in companies values. So this is crucial to do factor analysis before you run any algorithms. You can see an example of such analysis in ‘alpha_factor_analysis.ipynb’

For now on, we run several experiments with long/short strategies using Quantopian platform. We also used Genetic Algorithm for traditional approach as optimization for weights allocations. If you want to get acquainted with all details, check our poster in 'results/presentations' directory and presentation [here](https://docs.google.com/presentation/d/19knJVxAn4K7khZVep67Lgszupuyzt1Je9Mu0rliJmJ0/edit?usp=sharing). There are many visualizations and more detailed explanations.

## Some results

First round stands for model selection, we preserve the same time period and SPY index as performance benchmark. The second round stands for testing our algorithms, it consists of non-overlapping with first round time period. There are one long-short approach and the traditional one. It is worth adding that third long-short algorithm passed all the constraints to the Quantopian contest, you can find them [here](https://www.quantopian.com/contest). And, of course, we did factor analysis and run the backtest only with meaningful factors, which we know from alphalens framework.

![Results](https://github.com/adrianhryn/Portfolio-optimization/blob/master/results/results_table.png)

If you want to get acquainted with all details, check our poster in 'results/presentations' directory and presentation [here](https://docs.google.com/presentation/d/19knJVxAn4K7khZVep67Lgszupuyzt1Je9Mu0rliJmJ0/edit?usp=sharing). There are many visualizations and more detailed explanations.

## How to reproduce our work:

In our work, we used Quantopian API which you can not run outside of Quantopian environment.
So, in order to run our code you should sign up and log in [here](https://www.quantopian.com/posts)
Then you can upload our notebook and run it from top to down.

If you want to see the algorithms performance and their statistics, you should:
1. Click 'New Algorithm'
2. Upload a .py python file
3. Choose the backtest period
4. Click 'Build Algorithm'
5. Click 'Run Full Backtest'
6. Get the statistics

All the requirements to libraries versions are satisfied inside the Quantopian platform.
All the data you need you can query through the code using Quantopian API (which is implemented in our code).
Each of the .py files on this repository is a distinct algorithm which you can easily run if you follow the six steps above.


