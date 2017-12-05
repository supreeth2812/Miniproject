import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pandas.plotting import autocorrelation_plot


class PollFunc:
    def __init__(self, df):
        self.df = df
        self.sdf = pd.DataFrame

    def read_col(self, filename, colname):
        """
        Read concentration values from excel sheet
        colname = Column name in the df DataFrame for that pollutant
        """
        new_df = pd.read_excel(filename, skiprows=14, parse_dates=['Date'], dayfirst=True)
        if self.df.empty:
            self.df["Date"] = new_df['Date']
        self.df[colname] = new_df['Concentration']
        self.df = self.df.fillna(0)
        self.df = self.df[:-1]
        return self.df

    def find_aqi(self):
        """To find AQI that is max of sub-indices"""
        self.get_subindex()
        self.sdf = self.sdf.fillna(0)
        self.sdf["AQI"] = self.sdf[['PM', 'NO2', 'CO', 'SO2']].max(axis=1)

    def get_subindex(self):
        """To calculate subindex for each pollutant and store it into sdf DataFrame"""
        self.sdf = pd.DataFrame(columns=self.df.columns)
        self.sdf["Date"] = self.df["Date"]
        if 'PM' in self.df.columns:
            self.pm_subindex()
        if 'NO2' in self.df.columns:
            self.no2_subindex()
        if 'CO' in self.df.columns:
            self.co_subindex()
        if 'SO2' in self.df.columns:
            self.so2_subindex()

    def calculate_subindex(self, conc, blo, bhi, ilo, ihi):
        """General formula to calculate subindex of given concentration"""
        return (((ihi - ilo) / (bhi - blo)) * (conc - blo)) + ilo

    def pm_subindex(self):
        """To obtain subindex for PM pollutant according to the standards"""
        count = 0
        for i in self.df["PM"]:
            if int(i) in range(0, 30):
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 0, 30, 0, 50)
            elif int(i) in range(30, 60):
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 31, 60, 51, 100)
            elif int(i) in range(60, 90):
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 61, 90, 101, 200)
            elif int(i) in range(90, 120):
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 91, 120, 201, 300)
            elif int(i) in range(120, 250):
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 121, 250, 301, 400)
            elif int(i) > 250:
                self.sdf.at[count, 'PM'] = self.calculate_subindex(i, 251, 350, 401, 500)
            count += 1

    def no2_subindex(self):
        """To obtain subindex for NO2 pollutant according to the standards"""
        count = 0
        for i in self.df["NO2"]:
            if int(i) in range(0, 40):
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 0, 40, 0, 50)
            elif int(i) in range(40, 80):
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 41, 80, 51, 100)
            elif int(i) in range(80, 180):
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 81, 180, 101, 200)
            elif int(i) in range(180, 280):
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 181, 280, 201, 300)
            elif int(i) in range(280, 400):
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 281, 400, 301, 400)
            elif int(i) > 400:
                self.sdf.at[count, 'NO2'] = self.calculate_subindex(i, 401, 550, 401, 500)
            count += 1

    def co_subindex(self):
        """To obtain subindex for CO pollutant according to the standards"""
        count = 0
        for i in self.df["CO"]:
            if 0 <= i < 1.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 0, 1, 0, 50)
            elif 1.1 < i < 2.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 1.1, 2, 51, 100)
            elif 2.1 < i < 10.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 2.1, 10, 101, 200)
            elif 10.1 < i < 17.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 10.1, 17, 201, 300)
            elif 17.1 < i < 34.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 17.1, 34, 301, 400)
            elif i >= 34.1:
                self.sdf.at[count, 'CO'] = self.calculate_subindex(i, 34.1, 50, 401, 500)
            count += 1

    def so2_subindex(self):
        """To obtain subindex for SO2 pollutant according to the standards"""
        count = 0
        for i in self.df["SO2"]:
            if int(i) in range(0, 40):
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 0, 40, 0, 50)
            elif int(i) in range(40, 80):
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 41, 80, 51, 100)
            elif int(i) in range(80, 380):
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 81, 380, 101, 200)
            elif int(i) in range(380, 800):
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 381, 800, 201, 300)
            elif int(i) in range(800, 1600):
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 801, 1600, 301, 400)
            elif int(i) > 1600:
                self.sdf.at[count, 'SO2'] = self.calculate_subindex(i, 1601, 2000, 401, 500)
            count += 1

    def plot_aqi(self):
        """Plot AQI graph with hazard levels"""
        plt.plot(self.sdf["AQI"], color="royalblue")
        p = plt.axhspan(0, 50, 0, 700, facecolor='g', alpha=0.6)
        p = plt.axhspan(50, 100, 0, 700, facecolor='lightgreen', alpha=0.6)
        p = plt.axhspan(100, 200, 0, 700, facecolor='yellow', alpha=0.6)
        p = plt.axhspan(200, 300, 0, 700, facecolor='orange', alpha=0.6)
        p = plt.axhspan(300, 400, 0, 700, facecolor='r', alpha=0.6)
        p = plt.axhspan(400, 800, 0, 700, facecolor='firebrick', alpha=0.6)
        plt.axis([0, 700, 0, 800])
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title("AQI based on hazard levels")
        plt.show()

    def plot_correlation(self, df):
        """
        Plot correlation matrix to show correlation between different pollutants and
        meteorological parameters with AQI.
        """
        df["AQI"] = self.sdf["AQI"]
        df.set_index('Date', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        corr = df.corr()
        plt.matshow(df.corr())
        plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')
        plt.xticks(range(len(df.columns[0:250])), df.columns)
        plt.yticks(range(len(df.columns[0:250])), df.columns)
        plt.title('Correlation Matrix of all the factors in pollution analysis')
        plt.colorbar()
        plt.show()

    def plot_boxplot(self, x, title="Data"):
        """
        Plot boxplot for x
        x = Any pollutant column in DataFrame
        """
        plt.boxplot(x)
        plt.title('Box plot for ' + title)
        plt.show()

    def seasonal_analysis(self):
        """Provide seasonal analysis of each season of a year"""
        winter = self.sdf["AQI"]
        winter_avg = winter[88:204].mean()
        summer = self.sdf["AQI"]
        summer_avg = summer[205:266].mean()
        monsoon = self.sdf["AQI"]
        monsoon_avg = monsoon[267:311].mean()
        autumn = self.sdf["AQI"]
        autumnn_avg = autumn[312:392].mean()

        number_of_seasons = 4
        xvalues = np.arange(number_of_seasons)
        yvalues = (winter_avg, summer_avg, monsoon_avg, autumnn_avg)
        width = 0.3
        opacity = 0.8
        colors = ["skyblue", "orange", "lightslategray", "peru"]
        plt.xticks(xvalues, ('winter', 'summer', 'monsoon', 'autumn'))
        plt.xlabel('Seasons')
        plt.ylabel('AQI')
        plt.title('Seasonal AQI analysis for bangalore')
        plt.bar(xvalues, yvalues, width, alpha=opacity, label='Year 2015-16', align='center', color=colors)
        plt.tight_layout()
        plt.show()

    def test_stationarity(self, timeseries):
        """Statistical test to check if the timeseries is stationary"""

        # Determining rolling statistics
        rolmean = pd.rolling_mean(timeseries, window=12)
        rolstd = pd.rolling_std(timeseries, window=12)

        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()

    def regression_model(self):
        """Build Multiple linear regression model"""
        x = self.df[["CO", "PM", "SO2", "NO2"]]
        y = self.sdf["AQI"]
        date = np.arange(0, len(self.sdf["AQI"]), 1)
        date_train = np.arange(0, 300, 1)
        date_test = np.arange(300, 600, 1)
        X_train = x[:300]
        X_test = x[300:600]
        Y_train = y[:300]
        Y_test = y[300:600]

        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_train, Y_train)
        # Make predictions using the testing set
        Y_predict = regr.predict(X_test)

        print('Coefficients: \n', regr.coef_)
        print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_predict))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(Y_test, Y_predict))
        # Plot outputs
        plt.scatter(date_test, Y_predict, color='black')
        plt.plot(date, y, color="blue")
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title("Linear Regression Model Predictions")
        plt.show()

    def fit_ARIMA(self):
        """Build ARIMA model"""
        if type(self.sdf.index)is not 'pandas.core.indexes.datetimes.DatetimeIndex':
            self.sdf.set_index('Date', inplace=True)
            self.sdf.index = pd.DatetimeIndex(self.sdf.index)
        self.test_stationarity(self.sdf["AQI"])
        date = np.arange(0, len(self.sdf["AQI"]), 1)
        autocorrelation_plot(self.sdf["AQI"])
        plt.show()
        X = self.sdf["AQI"]
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)
        # plotting predictions against test data
        plt.plot(predictions)
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title("Predicted Values")
        plt.show()
        pred = len(predictions)
        length = len(train)
        plt.plot(date[:length], train, color="blue")
        plt.plot(date[length:length + len(test)], test, linestyle='--', color="blue")
        plt.plot(date[length:length + pred], predictions, color='red')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.title("AQI Forecast")
        blue_patch = mpatches.Patch(color='blue', label='Original')
        red_patch = mpatches.Patch(color='red', label='Predicted')
        plt.legend(handles=[blue_patch, red_patch])
        plt.show()

    def decomposition_ARIMA(self):
        """ARIMA model based on decomposition"""
        if type(self.sdf.index) is not 'pandas.core.indexes.datetimes.DatetimeIndex':
            self.sdf.set_index('Date', inplace=True)
            self.sdf.index = pd.DatetimeIndex(self.sdf.index)
        self.test_stationarity(self.sdf["AQI"])
        ts = self.sdf["AQI"]
        ts_log = np.log(ts)
        plt.plot(ts_log)
        decomposition = seasonal_decompose(ts_log, freq=30)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        plt.subplot(411)
        plt.plot(ts_log, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        # select residual to fit into model
        ts_log_decompose = residual
        ts_log_decompose.dropna(inplace=True)
        self.test_stationarity(ts_log_decompose)

        lag_acf = acf(ts_log_decompose, nlags=20)
        lag_pacf = pacf(ts_log_decompose, nlags=20, method='ols')
        # Plot ACF:
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(ts_log_decompose)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(ts_log_decompose)), linestyle='--', color='gray')
        plt.title('Autocorrelation Function')
        # Plot PACF:
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.axhline(y=-1.96 / np.sqrt(len(ts_log_decompose)), linestyle='--', color='gray')
        plt.axhline(y=1.96 / np.sqrt(len(ts_log_decompose)), linestyle='--', color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()

        # p=2 , q=2
        # ARIMA Model
        model = ARIMA(ts_log_decompose, order=(2, 0, 2))
        results_ARIMA = model.fit(disp=-1)
        plt.plot(ts_log_decompose)
        plt.plot(results_ARIMA.fittedvalues, color='red')
        plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_log_decompose) ** 2))
        plt.show()

        model = ARIMA(seasonal, order=(2, 0, 2))
        seasonal_results_ARIMA = model.fit(disp=-1)
        print(results_ARIMA.summary())
        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        print(predictions_ARIMA_diff.head())
        predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff, fill_value=0)
        print(predictions_ARIMA_log.head())
        predictions_s_ARIMA_diff = pd.Series(seasonal_results_ARIMA.fittedvalues, copy=True)
        print(predictions_s_ARIMA_diff.head())
        predictions_s_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
        predictions_s_ARIMA_log = predictions_s_ARIMA_log.add(predictions_s_ARIMA_diff, fill_value=0)
        print(predictions_s_ARIMA_log.head())
        predictions_ARIMA_log.add(predictions_s_ARIMA_log)
        print(predictions_ARIMA_log)
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        plt.plot(ts)
        plt.plot(predictions_ARIMA)
        plt.title('RMSE: %.4f' % np.sqrt(sum((predictions_ARIMA - ts) ** 2) / len(ts)))
        plt.show()