import PollutionFunc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
This is the python script where user has imported our PollutionFunc.py to utilize the available functions.
"""
peenya_df = pd.DataFrame()
peenya_pf = PollutionFunc.PollFunc(peenya_df)

peenya_df = peenya_pf.read_col("PEENYA\PM.xlsx", "PM")
peenya_df = peenya_pf.read_col("PEENYA\\NO2.xlsx", "NO2")
peenya_df = peenya_pf.read_col("PEENYA\CO.xlsx", "CO")
peenya_df = peenya_pf.read_col("PEENYA\SO2.xlsx", "SO2")

peenya_pf.find_aqi()

peenya_df = peenya_pf.read_col("PEENYA\\temp.xlsx", "Temp")
peenya_df = peenya_pf.read_col("PEENYA\\humidity.xlsx", "Humidity")
peenya_df = peenya_pf.read_col("PEENYA\\wind.xlsx", "Wind")

peenya_pf.plot_aqi()
peenya_pf.plot_correlation(peenya_df)
peenya_pf.seasonal_analysis()
peenya_pf.plot_boxplot(peenya_pf.sdf["AQI"], "AQI")
peenya_pf.plot_boxplot(peenya_pf.df["CO"])
peenya_pf.regression_model()
peenya_pf.decomposition_ARIMA()
peenya_pf.fit_ARIMA()

"""
This is a case study showing how the user can user our PollutionFunc.py functionality to perform
their very own analytics based on the AQI results generated. Here the user has performed Diwali
analysis for three years - 2015,16 and 17. 
"""
diwali_df = pd.DataFrame()
diwali_pf = PollutionFunc.PollFunc(diwali_df)

diwali_df = diwali_pf.read_col("PEENYA\diwali\\PM.xlsx", "PM")
diwali_df = diwali_pf.read_col("PEENYA\diwali\CO.xlsx", "CO")
diwali_df = diwali_pf.read_col("PEENYA\diwali\SO2.xlsx", "SO2")
diwali_df = diwali_pf.read_col("PEENYA\diwali\\NO2.xlsx", "NO2")

diwali_pf.find_aqi()

year_2015 = peenya_pf.sdf["AQI"]
diwali_2015 = year_2015[70:76].mean()
year_2016 = peenya_pf.sdf["AQI"]
diwali_2016 = year_2016[361:367].mean()
year_2017 = diwali_pf.sdf["AQI"]
diwali_2017 = year_2017[0:5].mean()
number_of_years = 3
xval = np.arange(number_of_years)
yval = (diwali_2015 - 50, diwali_2016 - 50, diwali_2017 - 50)
width = 0.3
opacity = 0.8
plt.xticks(xval, ('2015', '2016', '2017'))
plt.yticks([0, 25, 50, 100], [50, 75, 100, 150])
plt.xlabel('Monthly average')
plt.ylabel('AQI')
plt.title('Diwali AQI analysis for bangalore')
plt.bar(xval, yval, width, alpha=opacity, label='Year 2015-16')
plt.show()

"""
The same PollutionFunc.py can be used for data analysis for any area, any region, any country
given raw data regarding pollution levels recorded is available.
Below shows the AQI plot for BTM Layout data available.
"""
btm_df = pd.DataFrame()
btm_pf = PollutionFunc.PollFunc(btm_df)

btm_df = btm_pf.read_col("BTM\PM.xlsx", "PM")
btm_df = btm_pf.read_col("BTM\\NO2.xlsx", "NO2")
btm_df = btm_pf.read_col("BTM\CO.xlsx", "CO")
btm_df = btm_pf.read_col("BTM\SO2.xlsx", "SO2")

btm_pf.find_aqi()
#btm_pf.plot_aqi()
