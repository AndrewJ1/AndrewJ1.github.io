---
layout: post
title:      "Time Series. Minding your p’s and q’s."
date:       2019-10-28 17:31:11 +0000
permalink:  time_series_minding_your_p_s_and_q_s
---


**What is a time series?**

A times series is a data set where the values are measured over time. Time series analysis is getting meaningful information about what has happened from a series of data points in the past and attempting to predict what will happen in the future. Data can be spaced at regular intervals – daily, weekly, monthly … or could be more sporadic such as logging when certain events occurred. Time series is widely used in variety of fields.  Making predictions to better understand and plan for the future is useful almost everywhere.

In this blog I will attempt to simplify the components of a time series ARIMA model as much as possible, so that you can use it with full confidence. I will give some coding examples from a project I worked on using housing prices in Austin.

The first step is to determine whether a series is stationary. Stationarity means that the distribution of the data does not change with time. If the data are stationary statistical properties such as mean and variance remain constant over time. There is no trend or seasonality of the data.  

Why do we care about stationarity? Well, if the properties of the data vary over we end up with too many parameters to model. We could even end up with more parameters that we have data.


**Stationarity is important … so now what?**

Fortunately there is a useful test for checking stationarity – the Augmented Dickey Fuller (ADF) Test. This test has the starting assumption (or Null-hypothesis) that the time series is not stationary. We use the ADF to see if the data gives us a reason to reject the null hypothesis and say that the series is stationary.

Code for the ADF test. 

```
from statsmodels.tsa.stattools import adfuller
adfuller(df ['value'])

```

The p-value is the 2nd value returned. If it is less than 0.05, we can reject the null hypothesis and say that the series is stationary


Even if the data is not stationary all is not lost. There are several methods we can use to eliminate trends and make the data stationary. 
Applying a log transformation makes the time series more even over time. The advantage of taking the log is that higher values are penalized more than lower values. Log transformation works especially well for multiplicative trends. 
Another method is to calculate the rolling mean and subtract it from the time series, to make your time series is stationary
A third method of dealing with trend and seasonality is differencing. In this technique, we take the difference between an observation at a point in time with an observation at a previous time. The first differencing value is the difference between the current time period and the previous time period. If this difference does not give the data a constant mean and variance then we repeat the process, and find the second differencing starting with the values of the first differencing. Each difference step is called an order.
First order differencing can be done in Pandas using the .diff() function with periods = 1 (or 1-period lag).



One of the main ways to choose the right model is by using the ACF and PACF.

**Autocorrelation** (ACF) is the degree that a time series is related to a previous copy of itself. It helps us understand how each time series observation is related to previous time steps, which we call lags. Data with a greater autocorrelation is more predictable. Like differencing, each lag that we include is called an order.

**Partial Autocorrelation Function** (PACF) is a summary of the relationship between an element in a time series with observations at prior time steps (or lags), with the relationship of in between observations removed. PACF measures the incremental benefit of adding another lag to the model.
For example, if we consider the price of apples today, the PACF of the price of apples 3 months ago, would exclude the effect of the price of apple 1 month ago and 2 months ago.  
PACF contrasts with the autocorrelation function, which does not control for other lags. 

Code for how to plot the ACF and PACF
```
# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 14, 5

# Take first difference of the price
chg_price = df.diff(periods=1)
chg_price = chg_price.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_price, lags=20, ax=axes[0] )

# Plot the PACF
plot_pacf(chg_price, lags=20, ax=axes[1])
fig.tight_layout()
plt.show();

```


**Now for the p’s and q’s. Our first model**

**AR**(p)
An autoregressive (AR) model is when a value from a time series is regressed on previous values from the same time series.
Or in a word formula:
Today = constant + slope × yesterday + noise

An AR model uses the value “p” for the order of the model. You can think of p as the number of periods in the model. For example, an AR(1) would be a “first order” autoregressive model. The outcome in a first order AR model is related only to time periods that are one period apart. A second or third order AR model would be related to data two or three periods apart.

ACF and PACF help us identify when to use an AR model. The model is AR if the ACF trails off after a lag and has a hard cut-off in the PACF after a lag. This lag is taken as the value for p.




**MA**(q)
The Moving Average model can be described as the weighted sum of today's and yesterday's noise.
In a word formula:
Today = Mean + Noise + Slope × yesterday's noise

A MA model uses the value “q” for the order of the model. You can think of q as the quantity of the lag of noise, where noise is a part of the time series not explained by trend or seasonality.

Again, ACF and PACF help us identify when to use an MA model. The model is MA if the PACF trails off after a lag and has a hard cut-off in the ACF after the lag. This lag value is taken as the value for q.


Combining AR and MA gives us an ARMA model. As we had to difference our time series to make it stationary, this model is trained to predict the difference of the time series. However, what we really want to predict is the actual value of the time series. We can do this by rebuilding our original time series. We start by reversing the differencing – by taking the cumulative sum, or integral. This gives the value of how much the time series changed from its initial value over our forecast. To get the absolute value we add the last value of the original time series to this change. 

```
Mean_forecast = cumsum(diff_forecast) + df.iloc[-1,0]
```

The steps of differencing, fitting the ARMA model and integrating the forecast sound like a lot of work.  Fortunately there is a model which does this for us.  Introducing the ARIMA model or autoregressive integrated moving average model.

The ARIMA model has an additional parameter “d”.  D refers to the number of differencing transformations required by the time series to get stationary. The orders are run in the ARIMA model as (p,d,q).
The more levels of order in a model, the better the model will fit the data. However this can lead to a problem of overfitting. We wanted to difference our data the minimum possible amount, only until it is stationary and then no more. We can use the ADF test to determine the appropriate order of differencing.

The ARIMA model takes care of the differencing and integrating steps, and is a simpler way to get a forecast for a non-stationary time series. 

Hopefully this blog has helped you better understand the components of an ARIMA model so that you can now use it with confidence.

Code for running an ARIMA model

```
# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima_model import ARIMA

# Forecast prices using an ARIMA(1,1,1) model
mod = ARIMA(df, order=(1,1,1))
res = mod.fit()

print(res.summary())


```


