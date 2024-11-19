import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score

processed_data = pd.read_csv("revised_processed_data.csv")

naics_codes = pd.unique(processed_data["NAICS"])

predictions = []

for naics_code in naics_codes:
    data_naics = processed_data[processed_data["NAICS"] == naics_code]
    data_naics = data_naics.drop("NAICS", axis=1)
    actual_2018_2019 = data_naics[data_naics["YEAR"] >= 2018]
    actual_2018_2019 = actual_2018_2019[actual_2018_2019["YEAR"] <= 2019]
    actual_2018_2019 = actual_2018_2019.set_index(actual_2018_2019["YEAR"])
    data_naics = data_naics[data_naics["YEAR"] < 2018]
    data_naics = data_naics.set_index(data_naics["YEAR"])
    data_naics = data_naics.drop("YEAR", axis=1)
    data_naics_total_emp = data_naics["TOT_EMP"]
    data_naics = data_naics.drop("TOT_EMP", axis=1)
    data_naics_total_emp_pct = data_naics_total_emp.pct_change() * 100
    data_naics_total_emp_pct = data_naics_total_emp_pct.dropna()
    # drop 2005 since there is % change is NA since it is the first year in the data
    data_naics = data_naics.drop(2005)

    model = arch_model(data_naics_total_emp_pct, x=data_naics, vol='GARCH', p=1, q=1)

    res = model.fit(disp="off")
    forecasts = res.forecast(horizon=2)
    
    tot_emp_naics_2018 = (1 + (forecasts.mean.iloc[0].iloc[0] / 100)) * data_naics_total_emp[2017]
    tot_emp_naics_2019 = (1 + (forecasts.mean.iloc[0].iloc[1] / 100)) * tot_emp_naics_2018

    variance_2018 = forecasts.variance.iloc[0].iloc[0]
    variance_2019 = forecasts.variance.iloc[0].iloc[1]

    # 1 for increase, 0 for decrease
    pred_inc_or_dec_2018 = 1
    actual_inc_or_dec_2018 = 1
    pred_inc_or_dec_2019 = 1
    actual_inc_or_dec_2019 = 1

    if (forecasts.mean.iloc[0].iloc[0]) < 0:
        pred_inc_or_dec_2018 = 0
    
    if (actual_2018_2019["TOT_EMP"][2018] - data_naics_total_emp[2017] < 0):
        actual_inc_or_dec_2018 = 0

    if (forecasts.mean.iloc[0].iloc[1]) < 0:
        pred_inc_or_dec_2019 = 0
    
    if (actual_2018_2019["TOT_EMP"][2019] - actual_2018_2019["TOT_EMP"][2018] < 0):
        actual_inc_or_dec_2019 = 0

    predictions.append((naics_code,
                        (float(tot_emp_naics_2018), float(variance_2018), int(actual_2018_2019["TOT_EMP"][2018]), pred_inc_or_dec_2018, actual_inc_or_dec_2018),
                        (float(tot_emp_naics_2019), float(variance_2019), int(actual_2018_2019["TOT_EMP"][2019]), pred_inc_or_dec_2019, actual_inc_or_dec_2019)))


# Predictions format for each NAICS code:
# (naics_code, 
# (total employment prediction 2018, % variance for 2018 prediction, actual total employment 2018, predicted whether increased/decreased, actual whether increased/decreased),
# (total employment prediction 2019, % variance for 2019 prediction, actual total employment 2019, predicted whether increased/decreased, actual whether increased/decreased))
actual = pd.Series()
predicted = pd.Series()
actual_inc_or_dec = pd.Series()
pred_inc_or_dec = pd.Series()

for prediction in predictions:
    print(prediction)
    actual = np.append(actual, prediction[1][2])
    predicted = np.append(predicted, prediction[1][0])
    actual_inc_or_dec = np.append(actual_inc_or_dec, prediction[1][4])
    pred_inc_or_dec = np.append(pred_inc_or_dec, prediction[1][3])
    actual = np.append(actual, prediction[2][2])
    predicted = np.append(predicted, prediction[2][0])
    actual_inc_or_dec = np.append(actual_inc_or_dec, prediction[2][4])
    pred_inc_or_dec = np.append(pred_inc_or_dec, prediction[2][3])


print("------------------------")
print()
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
actual_min, actual_max = actual.min(), actual.max()
normalized_rmse = rmse / (actual_max - actual_min)
print("Normalized RMSE:", normalized_rmse)

r2 = r2_score(actual, predicted)
print("r2:", r2)

actual_inc_or_dec = actual_inc_or_dec.astype(int)
pred_inc_or_dec = pred_inc_or_dec.astype(int)
precision = precision_score(actual_inc_or_dec, pred_inc_or_dec)
recall = recall_score(actual_inc_or_dec, pred_inc_or_dec)
f1 = f1_score(actual_inc_or_dec, pred_inc_or_dec)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)