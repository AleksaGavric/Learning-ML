import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Read the CSV file into a DataFrame
df = pd.read_csv("honeyproduction.csv")

# Group the data by total production per year
prod_per_year = df.groupby("year")["totalprod"].sum().reset_index()

# Prepare the data for linear regression
X = prod_per_year["year"].values.reshape(-1, 1)
y = prod_per_year["totalprod"]

# Plot the original data
plt.scatter(X, y)

# Create a linear regression model and fit it to the data
regr = LinearRegression()
regr.fit(X, y)

# Generate predictions using the linear regression model
y_predict = regr.predict(X)

# Plot the predicted values
plt.scatter(X, y_predict)

# Generate predictions for future years
X_future = np.array(range(2013, 2051)).reshape(-1, 1)
future_predict = regr.predict(X_future)

# Plot the predicted values for future years
plt.scatter(X_future, future_predict)

# Show the plot
plt.show()
