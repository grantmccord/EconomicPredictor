import pandas as pd
from arch import arch_model

processed_data = pd.read_csv("revised_processed_data.csv")

naics_codes = pd.unique(processed_data["NAICS"])

predictions = []

for naics_code in naics_codes:
    data_naics = processed_data[processed_data["NAICS"] == naics_code]
    data_naics = data_naics.drop("NAICS", axis=1)
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

    predictions.append((naics_code, (float(tot_emp_naics_2018), float(variance_2018)), (float(tot_emp_naics_2019), float(variance_2019))))


# Predictions format for each NAICS code:
# (naics_code, (total employment prediction 2018, % variance for 2018 prediction), (total employment prediction 2019, % variance for 2019 prediction))
for prediction in predictions:
    print(prediction)