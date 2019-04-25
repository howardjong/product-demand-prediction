# product-demand-prediction
Various popular regression methods (Random Forest, Adam Optimization, and XGBoost) were compared for the prediction for the time-series unit demand of over 2000 products. The data is fairly large with > 1M rows, but relatively few features, which results in weak predictability at the demand extremes. In real business terms, the poor predictabiltiy of very large demands (>3000 units) means the model isn't helpful for the largest and most important orders. Nonetheless, the implemtation of this model serves as a reasonable predictor of demand for the majority of the unit orders with a Spearman rho value of 82%, which could facilitate supply chain management for most of the products. The flask API enables the code to be deployed and receive data posts via Postman.

Improved predictions are possible if the order sizes are split into small, medium, and large buckets that each use different models. However, the purpose was to evaluate how popular the popular regression models compared with one another on a dataset that mimics real-life scenarios.

Input of the made up test data {"Product_Code":1410, "Year":2015, "Month":5, "Day":19, "Dayofweek":1} results in a prediction of 3812 units for Product Code 1410 using the random forest regressor model.
