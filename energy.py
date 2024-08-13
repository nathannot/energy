import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set_style('darkgrid')

# Load and prepare data
df = pd.read_csv('01 renewable-share-energy.csv')
df.columns = ['Entity', 'Code', 'Year', 'Renewable']

# Sidebar for data exploration
st.sidebar.title('Data Exploration')
entity = st.sidebar.selectbox('Select Entity', df['Entity'].unique())
year_range = st.sidebar.slider('Select Year Range', min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(int(df['Year'].min()), int(df['Year'].max())))

# Filter data based on selections
filtered_df = df[(df['Entity'] == entity) & (df['Year'].between(year_range[0], year_range[1]))]
st.sidebar.subheader('Filtered Data')
st.sidebar.write(filtered_df)

# Fit SARIMAX model and forecast
aus = filtered_df
x = aus.Renewable
bm = SARIMAX(x, order=(2, 1, 2)).fit()  # Using original parameters
forecast_years = st.slider('Forecast Horizon (Years)', min_value=1, max_value=50, value=15)
forecast = bm.forecast(forecast_years)
fore = pd.DataFrame({'Forecasted %': forecast.round(2)})
fore['Year'] = range(aus.Year.max() + 1, aus.Year.max() + 1 + forecast_years)

# Plot
st.subheader('Forecast Plot')
fig, ax = plt.subplots()
ax.plot(aus.Year, x, label='Historical Data')
ax.plot(fore.Year, fore['Forecasted %'], label='Forecasted %', color='orange')
ax.set_xlim(aus.Year.min(), aus.Year.max() + forecast_years)
ax.set_xlabel('Year')
ax.set_ylabel('Renewable %')
ax.legend()
st.pyplot(fig)

# Forecast Table
st.subheader('Forecast Table')
st.write(fore)


