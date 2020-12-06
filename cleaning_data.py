import pandas as pd


price_data = pd.read_csv('dispatchprice_fy2009-2019.csv')
price_data = price_data[price_data['REGIONID'] == 'SA1']
price_data['SETTLEMENTDATE'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
price_data.index = price_data['SETTLEMENTDATE']
price_data.drop(columns='SETTLEMENTDATE', inplace=True)
columns_drop_price = ['RUNNO', 'INTERVENTION', 'EEP', 'ROP', 'APCFLAG', 'MARKETSUSPENDEDFLAG',
       'LASTCHANGED', 'RAISE6SECRRP', 'RAISE6SECROP', 'RAISE6SECAPCFLAG',
       'RAISE60SECRRP', 'RAISE60SECROP', 'RAISE60SECAPCFLAG', 'RAISE5MINRRP',
       'RAISE5MINROP', 'RAISE5MINAPCFLAG', 'RAISEREGRRP', 'RAISEREGROP',
       'RAISEREGAPCFLAG', 'LOWER6SECRRP', 'LOWER6SECROP', 'LOWER6SECAPCFLAG',
       'LOWER60SECRRP', 'LOWER60SECROP', 'LOWER60SECAPCFLAG', 'LOWER5MINRRP',
       'LOWER5MINROP', 'LOWER5MINAPCFLAG', 'LOWERREGRRP', 'LOWERREGROP',
       'LOWERREGAPCFLAG', 'PRICE_STATUS', 'PRE_AP_ENERGY_PRICE',
       'PRE_AP_RAISE6_PRICE', 'PRE_AP_RAISE60_PRICE', 'PRE_AP_RAISE5MIN_PRICE',
       'PRE_AP_RAISEREG_PRICE', 'PRE_AP_LOWER6_PRICE', 'PRE_AP_LOWER60_PRICE',
       'PRE_AP_LOWER5MIN_PRICE', 'PRE_AP_LOWERREG_PRICE',
       'CUMUL_PRE_AP_ENERGY_PRICE', 'CUMUL_PRE_AP_RAISE6_PRICE',
       'CUMUL_PRE_AP_RAISE60_PRICE', 'CUMUL_PRE_AP_RAISE5MIN_PRICE',
       'CUMUL_PRE_AP_RAISEREG_PRICE', 'CUMUL_PRE_AP_LOWER6_PRICE',
       'CUMUL_PRE_AP_LOWER60_PRICE', 'CUMUL_PRE_AP_LOWER5MIN_PRICE',
       'CUMUL_PRE_AP_LOWERREG_PRICE', 'OCD_STATUS', 'MII_STATUS', 'REGIONID', 'DISPATCHINTERVAL']
price_data = price_data.drop(columns_drop_price, axis=1)





region_data = pd.read_csv('dispatchregionsum_rrponly_fy2009-2019.csv')
region_data = region_data[region_data['REGIONID'] == 'SA1']
region_data['SETTLEMENTDATE'] = pd.to_datetime(region_data['SETTLEMENTDATE'])
region_data.index = region_data['SETTLEMENTDATE']
region_data.drop(columns='SETTLEMENTDATE', inplace=True)
columns_drop_region = ['RUNNO', 'INTERVENTION', 'REGIONID', 'DISPATCHINTERVAL']
region_data = region_data.drop(columns_drop_region, axis=1)



solar_data = pd.read_csv('wind_solar_demand.csv')
solar_data = solar_data[solar_data['REGIONID'] == 'SA1']
solar_data['SETTLEMENTDATE'] = pd.to_datetime(solar_data['SETTLEMENTDATE'])
solar_data = solar_data.drop([717364])
solar_data.index = solar_data['SETTLEMENTDATE']
solar_data.drop(columns=['SETTLEMENTDATE', 'TOTALDEMAND', 'REGIONID'], inplace=True)
solar_data = solar_data.resample('5min').fillna(method='ffill')







combined_data = pd.merge(region_data, price_data, left_index=True, right_index=True)
combined_data = pd.merge(combined_data, solar_data, left_index=True, right_index=True)
combined_data['TOTALDEMAND'].apply(lambda x: float(x))
combined_data['TOTALCLEARED_Solar'].apply(lambda x: float(x))
combined_data['TOTALCLEARED_Wind'].apply(lambda x: float(x))
"""
combined_data['ROOF_PV_POWER'] = combined_data['ROOF_PV_POWER'].str.lstrip()
combined_data['ROOF_PV_POWER'] = combined_data['ROOF_PV_POWER'].str.rstrip()
combined_data['ROOF_PV_POWER'] = combined_data['ROOF_PV_POWER'].str.replace("-0", "0")
combined_data['ROOF_PV_POWER'] = combined_data['ROOF_PV_POWER'].fillna("0")
combined_data['ROOF_PV_POWER'].apply(lambda x: float(x))
"""

combined_data['RESIDUAL_DEMAND'] = combined_data['TOTALDEMAND'] - (combined_data['TOTALCLEARED_Solar'] + combined_data['TOTALCLEARED_Wind'])
combined_data['RRP5MIN'] = combined_data['RRP'] / 12
combined_data.drop(columns=['TOTALDEMAND', 'TOTALCLEARED_Solar', 'TOTALCLEARED_Wind', 'ROOF_PV_POWER', 'RRP'] , inplace=True)

combined_data = combined_data[['RRP5MIN', 'RESIDUAL_DEMAND']]




combined_data.to_csv("combined_data.csv", index=True)
