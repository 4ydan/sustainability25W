#!/bin/bash

# Create issues for "Preprocessing and Data Engineering"
gh issue create --title "Data Loading" --body "load 100 random location files from the folder" --milestone "Preprocessing and Data Engineering"
gh issue create --title "Null Value Analysis" --body "analyse and discuss if and which values are missing in the data" --milestone "Preprocessing and Data Engineering"
gh issue create --title "Statistical Analysis" --body "basic data analysis like distributions, correlations... A simple statistical graph or summary table can be depicted here" --milestone "Preprocessing and Data Engineering"
gh issue create --title "Data Preprocessing" --body "impute missing values and transform/normalize any data, if needed" --milestone "Preprocessing and Data Engineering"

# Create issues for "Model Development and Analysis"
gh issue create --title "Forecasting Models" --body "Model: at least four types of models – time series / neural networks (NNs) Deep NNs (DNNs) / regression models and any other variants. Dataset: LamaH daily usage meteorological data. Develop the models, measure the accuracy for one day ahead. Compare your models with a naive baseline of your choise (e.g. MA, t-1, ARIMA, ...)" --milestone "Model Development and Analysis"
gh issue create --title "Hyperparameter Optimization" --body "Select the best performing model and then optimize hyperparameters of that model to study the best achievable results" --milestone "Model Development and Analysis"
gh issue create --title "Feature Importance Study" --body "perform a feature importance study, and report which features are significant in predicting the target variable" --milestone "Model Development and Analysis"
gh issue create --title "Feature Importance Visualization" --body "Different forecasting models have their own way for presenting feature importance: follow the model specific method and present the table or plots" --milestone "Model Development and Analysis"
gh issue create --title "Performance Metrics for Different Time Horizons" --body "Measure the performance metrics with different time horizons for your selected model: 1,3,7 day(s) ahead" --milestone "Model Development and Analysis"
