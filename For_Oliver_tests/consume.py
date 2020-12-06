import requests
import pandas as pd
import optimizer_module as om
import datetime
import FINAL_SUBMISSION as fs #price predictor script

base_url = 'http://127.0.0.1:8080'

# Starting battery charge
nxtBatCharge = 0

# Initial historical data -> converts csv file to list of dictionaries
historical_data = pd.read_csv('test_historical_data.csv', parse_dates=['SETTLEMENTDATE'])
historical_data['SETTLEMENTDATE'] = historical_data['SETTLEMENTDATE'].astype(str)
historical_data = historical_data.to_dict('records')

while True:
    response = requests.get(f'{base_url}/forecast')
    if response.status_code == 400:
        break

    input_data = response.json()
    # Adds new data to historical data list and removes oldest time -> always keeps 2017 entries
    historical_data.append(input_data)
    historical_data.pop(0)

    # Gets list of price predictions from predictor
    #price_predictions = price_predictor(last 2016 intervals) 
    price_predictions = [1,10,200,15,20,3,-5,2,500] # Random prices just to test everything is working
    
    # Inputs list of predictions into optimizer and outputs predNxtAction
    nxtAction, nxtBatCharge = om.optimize(price_predictions, nxtBatCharge)
    
    settlementDate = datetime.datetime.strptime(input_data["SETTLEMENTDATE"], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=5)
    
    # Outputs battery action and next charge to API
    output_data = {"Start_Time":str(settlementDate.strftime('%Y/%m/%d %H:%M')), "Battery_State":nxtAction}
    
    ### 

    response = requests.post(f'{base_url}/action', json=output_data)

