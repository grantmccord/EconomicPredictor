import pandas as pd
# Below lists start with index 0 = 2005, and index -1 = 2023
sp = [1178.28,
1290.01,
1511.14,
1403.22,
902.41,
1125.06,
1338.31,
1341.27,
1639.84,
1889.77,
2111.94,
2065.55,
2395.35,
2701.49,
2854.71,
2919.62,
4167.85,
4040.36,
4146.17
]
unemploymentvalues = [5.1,
4.6,
4.4,
5.4,
9.4,
9.6,
9.0,
8.2,
7.5,
6.3,
5.6,
4.8,
4.4,
3.8,
3.6,
13.2,
5.8,
3.6,
3.7
]
price_indexes = [107.9,
117.2,
118.6,
141.2,
116.8,
126.7,
143.1,
142.0,
139.4,
140.1,
126.5,
119.9,
122.7,
128.2,
127.0,
119.0,
132.8,
148.2,
139.7
]
cpi = [193.6,
201.3,
206.755,
215.208,
213.022,
217.290,
224.806,
228.713,
231.893,
236.918,
237.001,
239.557,
244.004,
250.792,
255.296,
255.848,
268.452,
291.359,
303.365
]
x = ''
alldata = pd.DataFrame()
for year in range(2005,2024):
    i = year-2005
    if year == 2014:
        x = 'x'
    df = pd.read_excel(f'blsdata/natsector_M{year}_dl.xls{x}')
    enddf = df[['OCC_CODE','TOT_EMP','EMP_PRSE','PCT_TOTAL','A_MEAN','MEAN_PRSE','A_PCT25','A_MEDIAN','A_PCT75']]
    newdf = enddf.drop(enddf[(enddf["A_MEAN"] == "*") | (enddf["EMP_PRSE"] == "**")].index)
    newdf = newdf.replace('#',239200)
    newdf['CPI'] = cpi[i]
    newdf['PRICE_INDEX'] = price_indexes[i]
    newdf['UNEMPLOYMENT_RATE'] = unemploymentvalues[i]
    newdf['SP_INDEX'] = sp[i]
    newdf['YEAR'] = year
    alldata = pd.concat([alldata,newdf],ignore_index=True)
    newdf.to_csv(f'processed_data/processed_{year}_data.csv', mode='w')
    print(f'Saved file in processeddata as processed_{year}_data.csv')
alldata.to_csv(f'processed_data/all_processed_data.csv', mode='w')