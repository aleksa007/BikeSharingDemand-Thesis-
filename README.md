# BikeSharingDemand-Thesis-
Predicting Bike Sharing demand with machine learning 


## Introduction

The objective of this research is to predict the number of bikes at station level for bike sharing
systems. Majority of previous work use computational models which are complex. An alternative
approach to predict the number of bikes is via machine learning. This thesis explores traditional
regression methods and autoregression. The auto-regression model provided the highest R2 with
the lowest RMSE among the algorithms. The most important variables were temperature and
time. Results of this research benefit all mobility share companies and their customers by enabling
them to use their vehicles more effectively and efficiently, without delay.

## Data

The dataset
was retrieved from UCI Machine Repository and contains the number of public bicycles. It is composed of 14 attributes: Date, Rented Bike Count, Hour, Temperature(°C),
Humidity (%), Wind Speed (m/s), Visibility (10m), Dew Point Temperature(°C), Solar
Radiation (MJ/m2), Rainfall (mm), Snowfall (cm), Seasons, Holidays, Functioning Day
with 8760 instances. Each 24 instances represent one day in a year. The target is the
number of bicycles rented per hour.

## Algorithms

To be able to achieve research goals, this research is trying to find best possible
regression algorithm in order to predict bike sharing demand count. For purpose of
this study 6 traditional regression algorithms and 1 times-series algorithm will be used,
namely Linear Regression (LR), Lasso and Ridge regression, Support Vector Regressor
6 (SVR), Decision trees (DT), Random Forest Regressor (RFR) and Autoregression (AR).
Figure 3 provides list of algorithms and their description

## Results

![image](https://user-images.githubusercontent.com/55929938/130474459-29b22e70-9d8c-42ce-9d55-5690641fbe6a.png)

## Future Work
To prevent unseen occurrences such as earthquake or global pandemic study
(Efendi et al., 2018) proposed fuzzy random auto-regression. The focus of their paper
is a triangular fuzzy number of data preparation technique for building an enhanced
fuzzy random auto-regression model for forecasting purposes using non-stationary
stock market data. Reflecting on their study bike count could be predicted on nonstationary level, data could be taken while the customers is riding the bike.
Besides machine learning algorithms this research could be extended towards deep
learning models. Deep learning models use neural network architectures that learn
features directly from the data without the need for manual feature extraction are used
to train deep learning models. Present research showed that non-linear models have
higher R2 and deep learning accounts for non-linear models.
14

Regarding only two variables that had the most importance on the models, in
future more variables could be added to check if the model accuracy will improve. But,
for the current data, data could be split in seasons where for each season predictive
models would be calculated. Data distributes bikes usage throughout the year, it shows
that data oscillates depending on season.This might produce better prediction then on
overall data.
The work presented in this paper suggests that future research should focus on
better representing data and its underlying structure using cutting-edge techniques
