import pandas as pd
import datetime as dt


wind_solar_data = pd.read_csv('wind_solar_demand.csv')
wind_solar_data = wind_solar_data[wind_solar_data['REGIONID'] == 'SA1']
wind_solar_data = wind_solar_data.drop([717364])
wind_solar_data = wind_solar_data.loc[wind_solar_data["SETTLEMENTDATE"] >= "2018-12-24 23"]
wind_solar_data = wind_solar_data.loc[wind_solar_data["SETTLEMENTDATE"] < "2019-07"]
wind_solar_data['SETTLEMENTDATE'] = pd.to_datetime(wind_solar_data['SETTLEMENTDATE'])
wind_solar_data.index = wind_solar_data['SETTLEMENTDATE']
wind_solar_data = wind_solar_data.drop(["SETTLEMENTDATE", "ROOF_PV_POWER", "REGIONID"], axis=1)
wind_solar_data = wind_solar_data.resample('5min').fillna(method='ffill')

combined_data = pd.read_csv('combined_data.csv')

combined_data = combined_data.loc[combined_data["SETTLEMENTDATE"] >= "2018-12-24 23:55"]
combined_data = combined_data.loc[combined_data["SETTLEMENTDATE"] < "2019-07"]
combined_data.index = combined_data['SETTLEMENTDATE']
combined_data = combined_data.drop(['SETTLEMENTDATE', "RESIDUAL_DEMAND"], axis=1)

input_test = pd.merge(combined_data, wind_solar_data, right_index=True, left_index=True)
columns_titles = ["RRP5MIN","TOTALDEMAND", "TOTALCLEARED_Wind", "TOTALCLEARED_Solar"]
input_test=input_test.reindex(columns=columns_titles)
input_test.columns = ['RRP5MIN', 'DEMAND', 'WIND', 'SOLAR']
historical_data_test = input_test.loc[input_test.index < "2019"]
input_test = input_test.loc[input_test.index >= "2019"]

input_test.index = input_test.index.strftime('%d/%m/%Y %H:%M')
historical_data_test.index = historical_data_test.index.strftime('%d/%m/%Y %H:%M')
input_test.index.names = ['SETTLEMENTDATE']
historical_data_test.index.names = ['SETTLEMENTDATE']

print(historical_data_test)
print(input_test)

input_test.to_csv("FINAL/test_input.csv")
historical_data_test.to_csv("FINAL/test_historical_data.csv")
