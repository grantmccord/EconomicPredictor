
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
import numpy as np

# Read in Data set

df = pd.read_csv("final_processed_data.csv")

# Drop columns that are deemed irrelevant using Best Subset algorithm and Multicollinearity comparisons

df = df.drop(columns=['A_MEDIAN', 'A_PCT75', 'A_PCT25', 'MEAN_PRSE', 'UNEMPLOYMENT_RATE', 'CPI']) 

# Train-test split by year within this OCC_CODE

train_data = df[(df['YEAR'] >= 2005) & (df['YEAR'] <= 2017)]
test_data = df[(df['YEAR'] >= 2018) & (df['YEAR'] <= 2019)]

# Features and target

X_train = train_data.drop(columns=['TOT_EMP', 'OCC_CODE'])
y_train = train_data['TOT_EMP']
X_test = test_data.drop(columns=['TOT_EMP', 'OCC_CODE'])
y_test = test_data['TOT_EMP']

# Model training

model = LinearRegression()
model.fit(X_train, y_train)

c = model.intercept_
m = model.coef_

# Predictions

y_pred = model.predict(X_test)
    
# Regression metrics

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
y_min, y_max = y_test.min(), y_test.max()
normalized_rmse = rmse / (y_max - y_min)
r2 = r2_score(y_test, y_pred)

# Classify growth (1) or decline (0) based on change in employment

actual_change = np.diff(y_test)
predicted_change = np.diff(y_pred)

actual_direction = np.where(actual_change > 0, 1, 0)
predicted_direction = np.where(predicted_change > 0, 1, 0)

# Classification metrics

precision = precision_score(actual_direction, predicted_direction)
recall = recall_score(actual_direction, predicted_direction)
f1 = f1_score(actual_direction, predicted_direction)


# Store results

results = {
        'Intercept': c,
        'Coefficients': m,
        'RMSE': rmse,
        'Normalized RMSE': normalized_rmse,
        'R2 Score': r2,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
}


# Print overall metrics of LR model 

for i in results: 
    print(i + ' = ' + str(results[i]))

