# Input prices into optimizer
import pandas as pd
import optimizer_module as om

model = "univariateLSTM"

results = pd.read_csv('FINAL_RESULTS_DATA/'+model+'_price_predictions.csv', parse_dates=['SETTLEMENTDATE'])
results2 = pd.read_csv('test_input.csv', parse_dates=['SETTLEMENTDATE'])
results["prediction"] = results["prediction"]*12
results["true values"] = results["true values"]*12
results2["true values"] = results2["RRP5MIN"]*12
results.index = results["SETTLEMENTDATE"]


numDays = len(results) # Number of days to run model
start = 0 # Starting time interval from price data
bStorage0 = 0 # Starting battery charge
intPerDay = 1

times = results[start:start+(numDays*intPerDay)].index.tolist()
predPrices = results.iloc[start:start+(numDays*intPerDay)]["prediction"].tolist()
realPrices = results2.iloc[start:start+(numDays*intPerDay)]["true values"].tolist()
outputResults = 1 # Tells optimizer_module to ouput results for each optimization
outputActions = 1 # Tells optimizer_module to ouput list of actions/states for battery use
outputCharge = 1 # Tells optimizer_module to ouput list of battery charges

print("Optimizing results for real prices")
realNxtAction, realNxtBatCharge, realActions, realBatCharge = om.optimize(realPrices, bStorage0, outputResults, outputActions, outputCharge)
realActions.append(0)
maxProfit = [0]
for j in range(1, len(results)):
    maxProfit.append(maxProfit[j-1] + realActions[j]*realPrices[j]/12)


print("\nOptimizing results for predicted prices\nProfit is incorrect as it is calculating predicted profit not actual profit")
predNxtAction, predNxtBatCharge, predActions, predBatCharge = om.optimize(predPrices, bStorage0, outputResults, outputActions, outputCharge)
predActions.append(0)
actualProfit = [0]
for j in range(1, len(results)):
    actualProfit.append(actualProfit[j-1] + predActions[j]*realPrices[j]/12)

print("\n----------RESULTS----------")
print("First prediction: "+str(times[0])+"\nFinal prediction: "+str(times[-1]))
print("Max profit possible: $%.2f" % (maxProfit[-1]))
print("Actual profit: $%.2f -> %.2f%% of optimal profit" % (actualProfit[-1],actualProfit[-1]/maxProfit[-1]*100))

predActions.pop(0), predBatCharge.pop(0), realActions.pop(0), realBatCharge.pop(0)

results = pd.DataFrame({'RRP5MIN': realPrices, 'ACTUALSTATE': predActions, 'CHARGE': predBatCharge, 'ACTUALPROFIT': actualProfit, 'OPTIMALSTATE': realActions, 'OPTIMALCHARGE': realBatCharge, 'OPTIMALPROFIT': maxProfit}, index=times)
results.index.name = "SETTLEMENTDATE"
results["RRP5MIN"] = results["RRP5MIN"]/12
results.to_csv("FINAL_RESULTS_DATA/"+model+"_report_results.csv")