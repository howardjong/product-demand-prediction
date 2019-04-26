# Comparing machine learning regression methods to predict product demand
Three popular regression methods (Random Forest, Adam Optimized Gradient Descent, and Bayesian Optimized XGBoost) were compared to predict the time-series unit demand of over 2000 products. The data is fairly large with > 1M rows, but relatively few key features, which results in weak predictability at the demand extremes for all models. In real business terms, the poor predictabiltiy of very large orders (>3000 units) means the models aren't helpful for the largest and most important orders. However, this dataset lacked customer information, which is typically available and would have improved the accuracy of the models. Nonetheless, the evaluation of these models serves as a helpful comparision of their performance for this type of data. To note, a basic flask API was also setup to enable model deployment and receives input data via Postman.

Overall, the Random Forest Regression method--the easiest model to perform--offered the best performance and was the least expensive computationally. Random Forest Regression achieved the highest rank correlation coefficients and a significantly lower average median error. As such, with some further optimization, Random Forest Regression could be considered to facilitate supply chain management for most of the products of moderate order size.

Improved predictions are also possible if the order sizes are split into small, medium, and large buckets that each use different models. However, further optimization was not necessary to fulfill the purpose of comparing the regression models.

# Results
Random Forest Regression:
Average absolute error: 5574.9 units,
Standard deviation:     28737.5 units,
Average median error:   420.0 units,
r2 score:               -0.041,
Kendall tau:  64.7%,
Spearman rho: 81.9%;

Adam Optimized Gradient Descent:
Average absolute error: 7114.8 units,
Standard deviation:     28290.8 units,
Average median error:   4333.3 units,
r2 score:               0.001,
Kendall tau:  11.5%,
Spearman rho: 17.1%;

Bayesian Optimized XGBoost:
Average absolute error: 5370.7 units,
Standard deviation:     26109.1 units,
Average median error:   995.3 units,
r2 score:               0.187,
Kendall tau:  51.5%,
Spearman rho: 70.3%.


