# Comparing machine learning regression methods to predict product demand
Three popular regression methods (Random Forest, Adam Optimized gradient descent, and Bayesian Optimized XGBoost) were compared to predict the time-series unit demand of over 2000 products. The data is fairly large with > 1M rows, but relatively few key features, which results in weak predictability at the demand extremes for all models. In real business terms, the poor predictabiltiy of very large orders (>3000 units) means the models aren't helpful for the largest and most important orders. However, this dataset lacked customer information, which is typically available and would have improved the accuracy of the models. Nonetheless, the implemtation of these models serve as a helpful comparision of their performance and were reasonable predictors of demand for the majority of the unit orders. A Spearman rho value of 82% was achieved by the Random Forest Regression method--the easiest model to perform--which could facilitate supply chain management for most of the products of moderate order size. A basic flask API was also setup to enable model deployment and receive input data via Postman.

Improved predictions are also possible if the order sizes are split into small, medium, and large buckets that each use different models. However, the purpose was to evaluate how the popular regression models compared with one another on a dataset that somewhat mimics real-life scenarios.

Input of the made up test data {"Product_Code":1410, "Year":2015, "Month":5, "Day":19, "Dayofweek":1} results in a prediction of 3812 units for Product Code 1410 using the random forest regressor model.
