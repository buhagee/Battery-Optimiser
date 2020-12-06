#!/usr/bin/env python

import gurobipy as gp
from gurobipy import GRB
from math import floor

# Useful Variable Definitions

# Num 5 min interval in an hour
hrIntervals = 12
# Num of decision variables in model
numDecisionVars = 7

# Default Paramters (Don't Change)

# Maximum discharge rate (MW)
maxDischargeRate = 1
# Maximum charge rate (MW)
maxChargeRate = -1
# Storage capacity of battery (MWh)
bStorage = 2
# Round Trip Efficiency of battery (unitless)
roundTripEfficiency = 0.81


def optimize(prices, bStorage0, outputStats=0, outputActions=0, outputCharge=0):
    # Input Paramters

    # bstorage0 -> Integer of starting battery charge (MWh)

    # prices -> List of price predictions for energy ($/MWh)

    try:
        # Create a new model
        m = gp.Model("batOptimser")

        # Create variables
        dispatchGen = {}
        dischargeBattery = {}
        chargeBattery = {}
        energyInStorage = {}
        dispatchCost = {}
        dispatchRevenue = {}
        dispatchProfit = {}

        for i, p in enumerate(prices):
            # Decision Variables
            dispatchGen[i] = m.addVar(lb=-GRB.INFINITY, name="dispatchGen")
            dischargeBattery[i] = m.addVar(lb=0, ub=maxDischargeRate, name="dischargeBattery")
            chargeBattery[i] = m.addVar(lb=maxChargeRate, ub=0, name="chargeBattery")
            energyInStorage[i] = m.addVar(lb=0, ub=bStorage, name="energyInStorage")

            dispatchCost[i] = m.addVar(lb=-GRB.INFINITY, name="dispatchCost")
            dispatchRevenue[i] = m.addVar(lb=-GRB.INFINITY, name="dispatchRevenue")
            dispatchProfit[i] = m.addVar(lb=-GRB.INFINITY, name="dispatchProfit",
                                         obj=-1)  # Objective Function (negative to maximize)

            # Constraints
            if (i > 0):
                preserveEnergyInStorage = m.addConstr(energyInStorage[i] == (energyInStorage[i - 1]) - (
                            roundTripEfficiency * chargeBattery[i - 1] / 12) - (dischargeBattery[i - 1] / 12))

            batteryDispatchDefinition = m.addConstr(dispatchGen[i] == dischargeBattery[i] + chargeBattery[i])

            dispatchCostDefinition = m.addConstr(dispatchCost[i] == -1 * chargeBattery[i] * p / 12)
            dispatchRevenueDefinition = m.addConstr(dispatchRevenue[i] == dischargeBattery[i] * p / 12)
            dispatchProfitDefinition = m.addConstr(dispatchProfit[i] == dispatchRevenue[i] - dispatchCost[i])

        # More Constraints
        EnergyInStorageCapacity0 = m.addConstr(energyInStorage[0] == bStorage0)

        # Preserve Energy In Storage Definition for final price (misses it in for loop)
        energyInStorage[len(prices)] = m.addVar(lb=0, ub=bStorage, name="energyInStorage")
        preserveEnergyInStorage = m.addConstr(energyInStorage[len(prices)] == (energyInStorage[len(prices) - 1]) - (roundTripEfficiency * chargeBattery[len(prices) - 1] / 12) - (dischargeBattery[len(prices) - 1] / 12))

        # Optimize model
        m.setParam('OutputFlag', False)  # Suppresses Gurobi output
        m.optimize()

        # Stats + Debugging

        profit = 0
        revenue = 0
        cost = 0
        charges = 0
        chargeMW = 0
        discharges = 0
        dischargeMW = 0
        waiting = 0
        interval = -1
        actions = []
        batCharges = []

        for i, v in enumerate(m.getVars()):
            if (i % numDecisionVars == 0):
                # Counts current time interval
                interval += 1

            if (v.varName == "energyInStorage" and interval == 1):
                # Gets next battery charge based on current optimal action
                nxtBatCharge = v.x

            if (v.varName == "dispatchGen"):
                # Model data collection
                if (v.x > 0):
                    if (interval == 0):
                        nxtAction = v.x
                    currAction = "DISCHARGE"
                    profit += v.x * prices[floor(i / numDecisionVars)] / hrIntervals
                    revenue += v.x * prices[floor(i / numDecisionVars)] / hrIntervals
                    discharges += 1
                    dischargeMW += v.x / 12
                elif (v.x < 0):
                    if (interval == 0):
                        nxtAction = v.x
                    currAction = "CHARGE"
                    profit += v.x * prices[floor(i / numDecisionVars)] / hrIntervals
                    cost += -v.x * prices[floor(i / numDecisionVars)] / hrIntervals
                    charges += 1
                    chargeMW += -v.x / 12
                else:
                    if (interval == 0):
                        nxtAction = v.x
                    currAction = "DO NOTHING"
                    waiting += 1

                actions.append(v.x)
                
            if (v.varName == "energyInStorage"):
                batCharges.append(v.x)

                # DEBUGGING FOR DECISION VARIABLES -> Iteration number, price and action assigned
                # print("\nPrice Interval %g, Price: %.4g\nAction = %s" % (interval+1, prices[interval], currAction))

            # DEBUGGING FOR DECISIONS VARIABLES -> Variable names and values assigned
            # print('%s %g' % (v.varName, v.x))

        # OPTIMAL SOLUTION STATS

        if (outputStats == 1):
            print('\nModel objective value: %.2f' % m.objVal)
            print("Predicted profit = $%.2f -> %.2f%% Profit" % (profit, 100 * profit / cost))
            print("Charged %.2f MW over %d charge intervals (Lost %.2f MW due to battery inefficiency)" % (
            chargeMW, charges, chargeMW - (roundTripEfficiency * chargeMW)))
            print("Total Cost $%.2f" % (cost))
            print("Discharged %.2f MW over %s discharge intervals" % (dischargeMW, discharges))
            print("Total Revenue $%.2f" % (revenue))
            print("Did nothing during %d time intervals" % (waiting))
            print("Min price: $%.2f, Max price: $%.2f" % (min(prices), max(prices)))


        # nxtAction = Optimal solution for next battery action (MWh)
        # CHARGE if < 0, DISCHARGE if > 0, DO NOTHING if = 0
        # nxtBatCharge = Bat charge next time interval if nxtAction is used and battery inefficiency accounted for

        if (outputActions == 1 and outputCharge == 1):
            return nxtAction, nxtBatCharge, actions, batCharges
        elif (outputActions == 1):
            return nxtAction, nxtBatCharge, actions
        elif (outputCharge == 1):
            return nxtAction, nxtBatCharge, batCharges
        else:
            return nxtAction, nxtBatCharge


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')
