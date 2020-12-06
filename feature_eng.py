import pandas as pd
from datetime import datetime

def subtract_years(dt, years):
    try:
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years)
    except ValueError:
        dt = str(dt)
        dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(year=dt.year-years, day=dt.day-1)
    return dt

data = pd.read_csv('combined_data.csv')

data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'])
data.index = data['SETTLEMENTDATE']
data.drop(columns='SETTLEMENTDATE', inplace=True)

def avg_year_price(df, date, num_years):
    total_sum = 0
    #print(f"THE DATE IS:{date}")
    for i in range(num_years+1):
        if date.year < 2017:
            break
        date = subtract_years(str(date), i)
        total_sum += df.loc[date]['RRP5MIN']
        #print(total_sum)
    #print(f'FINAL TOTAL SUM IS: {total_sum}')
    avg_price = total_sum / num_years
    return avg_price

def avg_year_demand(df, date, num_years):
    total_sum = 0
    #print(f"THE DATE IS:{date}")
    for i in range(num_years+1):
        if date.year < 2017:
            break
        date = subtract_years(str(date), i)
        total_sum += df.loc[date]['RESIDUAL_DEMAND']
        #print(total_sum)
    #print(f'FINAL TOTAL SUM IS: {total_sum}')
    avg_demand = total_sum / num_years
    return avg_demand





data['avg_price'] = data.index.to_series().apply(lambda x: avg_year_price(data, x, 3))
data['avg_demand'] = data.index.to_series().apply(lambda x: avg_year_demand(data, x, 3))


print(data.tail())

data.to_csv("combined_data.csv", index=True)




