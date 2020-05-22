# Welcome!

Hello and thanks for stopping by! 

My name is Judy Tsao. I have a Ph.D. in chemistry and a Master in Management Analytics. I have worked in science outreach, e-learning, and the consumer products industry, but I am interested in anything related to technology and operations. I have a passion for translating technical knowledge into tangible and actionable insights. 

This GitHub page documents of my journey to learning more modern data science techniques. The projects shown here are currently coding practices done at my leisure time to familiarize myself with different Python data analytics and visualization packages. I sorted the projects into different contexts/industries and each project contains different topics that I was learning at the time.

Most of the code were documented using Jupyter Notebook to display the plots and model results. More detailed breakdown of the projects can be found in the notebooks of each of these projects.

## Retail and eCommerce

### Store Items Demand Forecasts

**Topics and Tools**: time-series analysis, seasonality decomposition, ARIMA, Plotly
For any retailer, demand forecsting is important for proper planning and inventory management. This project details the use of the ARIMA model to conduct time-series prediction on a set of retail items. Plotly was used again to generate interactive visualization. Details can be found in the Jupyter Notebook link below.

[Jupyter Notebook](https://github.com/tsaofa/Projects/blob/master/Demand%20Forecasting/Store%20Demand%20Forecasting.ipynb)


### eCommerce Retail Data Analysis

The key to growing a business lies in understanding its consumers. Similar to the above analysis, the dataset used in this project was transactional data on an eCommerce site. This analysis, however, focused on customer-level data analytics. Topics explored here include: customer lifetime value, purchase timing prediciton, and multiclass prediciton using XGBoost.

[Jupyter Notebook](https://github.com/tsaofa/Projects/blob/master/eCommerce%20Analysis/eCommerce%20Analysis.ipynb)

## Tourism and Travel

### Flight Analytics

Are you ever curious why flights are delayed, and which airlines are the most reliable? This analysis explored the different causes to a delay in flights using a variety of data visualization. Additionally, it showcases the use of LightGBM in a classification problem and attempts to explain the results using SHAP values.

[Jupyter Notebook](https://github.com/tsaofa/Projects/blob/master/Flight%20Analytics/Flight%20Analytics.ipynb)

### AirBnB Bookings

[Tableau Dashboard](https://public.tableau.com/profile/judy.tsao#!/vizhome/BuenosAiresAirBnB/MainPage)

## Finance

### Company Earnings Prediction

**Topics and Tools**: WRDS database, earnings prediction, object-oriented programming
If you work for an investment bank, chances are you spend most of your time trying to predict earnings for a firm. But is it possible to do this automatically using a firm's historical data? Unlike the previous projects, this project was completed as part of my Masters degree. The code uses data from WRDS database to predict company earnings using a number of different earnings prediction models. The original goal was to attempt to reproduce some of the finidngs in academic literature (more details found in the code itself). This project, however, also enabled me to have the opportunity to write a wide variety of functions to make the code resuable for various datasets.

[Python Code](https://github.com/tsaofa/Projects/blob/master/Earnings%20Prediction/AccountingProject.py)


