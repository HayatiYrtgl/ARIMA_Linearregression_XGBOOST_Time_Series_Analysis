
### Importing Libraries
The code imports necessary libraries such as Pandas (`pd`), NumPy (`np`), Matplotlib (`plt`), and Seaborn (`sns`).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Loading Data
The code reads data from an Excel file located at "../dataset/proje_verileri.xlsx" into a Pandas DataFrame named `data`.

```python
data_path = "../dataset/proje_verileri.xlsx"
data = pd.read_excel(data_path)
```

### Data Exploration
- `data.head()` and `data.tail()` display the first and last few rows of the DataFrame, respectively.
- `data.describe()` provides descriptive statistics for the numeric columns in the DataFrame.
- `data.info()` prints information about the DataFrame, including the data types and number of non-null values.

```python
data.head()
data.tail()
data.describe()
data.info()
```

### Data Cleaning
- `data.isna().sum()` checks for missing values in the DataFrame and prints the sum of missing values for each column.
- `data = data.sort_values(by=["Year", "Month"])` sorts the DataFrame by the "Year" and "Month" columns.
- Concatenates the "Year" and "Month" columns into a new column named "Date" as strings.
- Drops the "Year" and "Month" columns from the DataFrame.
- Drops the "CenterID" column from the DataFrame.
- Converts the "Date" column to datetime data type.
- Converts the "Date" column values to the format 'YYYY-MM'.
- Sets the "Date" column as the index of the DataFrame.

```python
data.isna().sum()
data = data.sort_values(by=["Year", "Month"])
data["Date"] = data["Year"].astype(str) + "-" +data["Month"].astype(str)
data.drop(["Year", "Month"], axis=1, inplace=True)
data.drop("CenterID", axis=1, inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y-%m'))
data.set_index("Date", inplace=True)
```

### Data Visualization
- Creates a bar plot showing the counts of two categories ("A" and "B").
- Sets the title of the plot.

```python
A = data.Type.value_counts().iloc[0]
B = data.Type.value_counts().iloc[1]
plt.bar(["A", "B"],[A, B], color="red")
plt.title("COUNT PLOT")
```

### Calculating Means
- Calculates the mean values of two categories ("A" and "B") in the DataFrame.

```python
mean_A = data.groupby("Type").mean().iloc[0].values[0]
mean_B = data.groupby("Type").mean().iloc[1].values[0]
```

1. **One-Hot Encoding**:
   - `dummy = pd.get_dummies(data=data, columns=["Type"], drop_first=True, dtype=int)` performs one-hot encoding on the categorical column "Type" in the DataFrame `data`. It creates dummy variables for each category, dropping the first one to avoid multicollinearity issues. The resulting DataFrame is stored in `dummy`.

2. **Correlation Heatmap**:
   - `sns.heatmap(dummy.corr(), annot=True)` generates a heatmap showing the correlation between different variables in the `dummy` DataFrame.

3. **Time Series Analysis with ARIMA Model**:
   - `model = ARIMA(dummy.NumbUsed, order=(5, 1, 0))` initializes an ARIMA (AutoRegressive Integrated Moving Average) model with parameters `(p=5, d=1, q=0)`. 
   - `model_fit = model.fit()` fits the ARIMA model to the time series data.
   - `print(model_fit.summary())` prints a summary of the fitted ARIMA model.
   - `residuals = pd.DataFrame(model_fit.resid)` creates a DataFrame containing the residuals (errors) of the fitted ARIMA model.
   - `residuals.plot()` and `residuals.plot(kind='kde')` visualize the residuals as a line plot and density plot, respectively.
   - `print(residuals.describe())` prints summary statistics of the residuals.

4. **Walk-Forward Validation**:
   - The code performs walk-forward validation for the ARIMA model. It iteratively forecasts the next value in the time series and compares it to the actual value.
   - RMSE (Root Mean Squared Error) is calculated to evaluate the performance of the forecasts.

5. **Linear Regression and XGBoost Model**:
   - Linear Regression and XGBoost models are trained and evaluated on the dataset.
   - Mean squared error (RMSE) is calculated for both training and test sets for each model.

6. **Model Comparison**:
   - `scores = pd.DataFrame({"ARIMA-RMSE": rmse, "Linear Model-rmse": None}, index=[0])` initializes a DataFrame to store the RMSE scores of different models.
   - RMSE scores for Linear Regression and XGBoost models are added to the DataFrame.
   - `scores.plot(kind="bar")` generates a bar plot comparing the RMSE scores of different models.

This code block demonstrates various aspects of time series analysis, including data preprocessing, model fitting, evaluation, and comparison. Additionally, it utilizes machine learning techniques such as linear regression and XGBoost for comparison with the ARIMA model.
