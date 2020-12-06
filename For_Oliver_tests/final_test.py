import requests
import pandas as pd
import optimizer_module as om
from datetime import datetime
from datetime import timedelta
from math import floor
#import FINAL_SUBMISSION as fs #price predictor script

base_url = 'http://127.0.0.1:8080'

# Starting battery charge
nxtBatCharge = 0

# Initial historical data -> converts csv file to list of dictionaries
historical_data = pd.read_csv('test_historical_data.csv', parse_dates=['SETTLEMENTDATE'])
historical_data['SETTLEMENTDATE'] = historical_data['SETTLEMENTDATE'].astype(str)
historical_data = historical_data.to_dict('records')

# Tracks Battery
realProfit = [0]
realPrices = []
predActions = []
predBatCharge = [0] 
times = []

i = 0

while True:
    response = requests.get(f'{base_url}/forecast')
    if response.status_code == 400:
        break
        
    input_data = response.json()
    # Adds new data to historical data list and removes oldest time -> always keeps 2017 entries
    historical_data.append(input_data)
    historical_data.pop(0)

    # Gets list of price predictions from predictor
    #price_predictions = fs.feature_engineering_input(historical_data) 
    price_predictions = [1,10,200,15,20,3,-5,2,500] # Random prices just to test everything working
    
    if (i > 0):
        # Tracks real profit and real prices
        realProfit.append(realProfit[i-1] + action*input_data["RRP5MIN"])
        realPrices.append(input_data["RRP5MIN"])
        
        
    # Inputs list of predictions into optimiser and outputs predNxtAction
    action, nxtBatCharge = om.optimize(price_predictions, nxtBatCharge)
    predActions.append(action)
    predBatCharge.append(nxtBatCharge)
    
    # Gets next time interval which is the prediction interval
    settlementDate = datetime.strptime(input_data["SETTLEMENTDATE"], '%Y-%m-%d %H:%M:%S') + timedelta(minutes=5)
    times.append(settlementDate.strftime('%Y/%m/%d %H:%M'))
    
    # Outputs time interval and battery action to API
    output_data = {"Start_Time":str(times[-1]), "Battery_State":action}
    
    ###
    if (i % 288 == 0):
        print("Running day %g..." % (floor(i/(287))+1))
    i += 1
    response = requests.post(f'{base_url}/action', json=output_data)

times.pop()
# Optimizes battery for real prices and calculates optimal profit
realNxtAction, realNxtBatCharge, realActions, realBatCharge = om.optimize(realPrices, 0, 0, 1, 1)
realActions.append(0)
maxProfit = [0]
for j in range(1, i-1):
    maxProfit.append(maxProfit[j-1] + realActions[j]*realPrices[j])


print("\n----------RESULTS----------")
print("First prediction: "+str(times[0])+"\nFinal prediction: "+str(times[-1]))
print("Max profit possible: $%.4g" % (maxProfit[-1]))
print("Actual profit: $%.4g -> %.4g%% of optimal profit" % (realProfit[-1],realProfit[-1]/maxProfit[-1]*100))

# Outputs results to csv
predBatCharge.pop(), predActions.pop(), realProfit.pop(), realActions.pop(0), realBatCharge.pop(0)
predBatCharge.pop()
results = pd.DataFrame({'RRP5MIN': realPrices, 'ACTUALSTATE': predActions, 'CHARGE': predBatCharge, 'ACTUALPROFIT': realProfit, 'OPTIMALSTATE': realActions, 'OPTIMALCHARGE': realBatCharge, 'OPTIMALPROFIT': maxProfit}, index=times)
results.index.name = "SETTLEMENTDATE"
results.to_csv("FINAL_RESULTS.csv")
