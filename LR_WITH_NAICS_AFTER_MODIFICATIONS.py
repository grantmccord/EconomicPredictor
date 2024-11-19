
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
import numpy as np

# reading csv file 

df = pd.read_csv("revised_processed_data.csv")

# Drop columns that are deemed irrelevant using Best Subset algorithm and Multicollinearity comparisons

df = df.drop(columns=['A_MEDIAN', 'PRICE_INDEX', 'UNEMPLOYMENT_RATE', 'SP_INDEX']) 

#standardizing predictor variable columns: 

df_x = df[['EMP_PRSE', 'A_MEAN', 'MEAN_PRSE', 'A_PCT25', 'A_PCT75', 'CPI']]
df[['EMP_PRSE', 'A_MEAN', 'MEAN_PRSE', 'A_PCT25', 'A_PCT75', 'CPI']] = (df_x-df_x.mean())/df_x.std()

results = {}  # Store metrics for each NAICS

for NAICS in sorted(df['NAICS'].unique()):

    # Filter data for the current NAICS
    sector_data = df[df['NAICS'] == NAICS]
    
    # Train-test split by year within this NAICS

    train_data = sector_data[(sector_data['YEAR'] >= 2005) & (sector_data['YEAR'] <= 2017)]
    test_data = sector_data[(sector_data['YEAR'] >= 2018) & (sector_data['YEAR'] <= 2019)]

    # Features and target

    X_train = train_data.drop(columns=['TOT_EMP', 'NAICS'])
    y_train = train_data['TOT_EMP']
    X_test = test_data.drop(columns=['TOT_EMP', 'NAICS'])
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

    results[NAICS] = {
        'Intercept': c,
        'Coefficients': m,
        'RMSE': rmse,
        'Normalized RMSE': normalized_rmse,
        'R2 Score': r2,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }


# Print overall metrics per NAICS
for NAICS, metrics in results.items():
    print(f"NAICS {NAICS}:\n")
    print(f"  Intercept =  {metrics['Intercept']}")
    print(f"  Coefficients =  {metrics['Coefficients']}")
    print(f"  RMSE = {metrics['RMSE']}")
    print(f"  Normalized RMSE = {metrics['Normalized RMSE']}")
    print(f"  R2 Score = {metrics['R2 Score']}")
    print(f"  Precision = {metrics['Precision']}")
    print(f"  Recall = {metrics['Recall']}")
    print(f"  F1 Score = {metrics['F1 Score']}\n")

Normalized_RMSE = 0
R2_score = 0
Precision = 0
Recall = 0
F1_score = 0
counter = 0

for NAICS, metrics in results.items():

    Normalized_RMSE =  Normalized_RMSE + metrics['Normalized RMSE']
    R2_score = R2_score + metrics['R2 Score']
    Precision = Precision + metrics['Precision']
    Recall = Recall + metrics['Recall']
    F1_score = F1_score + metrics['F1 Score']
    counter = counter + 1

print('Avg Values across all NAICSs:')
print(f"  Normalized RMSE = {Normalized_RMSE/counter}")
print(f"  R2 Score = {R2_score/counter}")
print(f"  Precision = {Precision/counter}")
print(f"  Recall = {Recall/counter}")
print(f"  F1 Score = {F1_score/counter}\n")

