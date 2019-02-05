import xlsxwriter
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

debug = False

def static_forecast(xlsx,dataset):
  static_forecast = xlsx.add_worksheet('static_foreast')
  data = None
  period = []
  demand = []
  des_x = []
  des_demand = []
  reg_demand = []
  sea_factors = []
  avg_sea = []
  res_demand = []
  
  # our data 
  data = pd.read_csv(dataset)
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])
  
  # column names
  c_names = ['Year','Period','Demand','De-Seasonalized Demand','Regressed,Deseasonalized Demand', 'Seasonal Factors', 'Average Seasonal Factors', 'Reintroduce Seasonality']
  row = 0 
  col = 0
  for name in (c_names):
    static_forecast.write (row,col, name)
    col += 1
  
  # fill out year and period columns
  row = 1 
  col = 0
  for x in range (num_demand):
    mod = x%p
    quarter = p / 4
    if (mod == 0):
      static_forecast.write (row,col, (x/p)+1)
    static_forecast.write  (row,col+1, x+1)
    row += 1
  
  # Demand Data
  row = 1 
  col = 2
  for x in range (num_demand):
    static_forecast.write (row,col,demand[x]) 
    row += 1
  
  # Demand data, deseasonalized
  if (p % 2 == 0): 
    min_val= int((p/2) + 1)
    max_val= int(num_demand - (p/2) + 1)
    row = min_val
    col = 3
    for x in range (min_val,max_val):
      # extra -1 for array indices
      lower = int(x-(p/2))-1
      upper = int(x+(p/2))-1
      des=(demand[lower] + demand[upper]+(2 * np.sum(demand[lower+1:upper])))/(2*p)
      des_x.append(x)
      des_demand.append(des)
      static_forecast.write (row,col,des) 
      row += 1
  else:
    min_val = int (((p-1)/2)+1)
    max_val = int (num_demand - ((p-1)/2)+1) 
    row = min_val
    col = 3
    for x in range (min_val,max_val):
      lower = int (x-((p-1)/2)) 
      upper = int (x+((p-1)/2)) 
      des = (np.sum(demand[lower-1:upper]))/p
      des_x.append(x)
      des_demand.append(des)
      static_forecast.write (row,col,des)
      row += 1
  
  # Demand data, deseasonalized and regressed
  des_x = np.array(des_x)
  des_demand = np.array(des_demand)
  slope, intercept, r_value, p_value, std_err = stats.linregress(des_x,des_demand)
  row = 1
  col = 4
  for x in range (num_period):
    reg = intercept + ((x+1) * (slope))
    reg_demand.append (reg)  
    static_forecast.write (row,col,reg)
    row += 1
  
  # seasonal factors  
  row = 1
  col = 5
  for x in range (num_demand):
    sea = (demand[x] / reg_demand[x])  
    sea_factors.append (sea)
    static_forecast.write (row,col,sea)
    row += 1

  # average seasonal factors 
  row = 1
  col = 6
  for x in range (p):
    asea = (sea_factors[x] + sea_factors [x+p])/2
    avg_sea.append (asea) 
    static_forecast.write (row,col,asea)
    row += 1
  
  # reseasonalize demand
  row = 1
  col = 7
  for x in range (num_period):
    resea =  avg_sea[x%p] * reg_demand[x]
    res_demand.append (resea)
    static_forecast.write (row,col,resea)
    row += 1
  
  # plot regressed, deseasonalized demand and reseasonalized demand
  plt.plot (period,reg_demand,period,res_demand)
  plt.ylabel('demand')
  plt.xlabel('period')
  plt.show()

  # debug print statements
  if (debug):
    print ()
    print ("DESEASONALIZED DEMAND: [x] , [DEMAND]")
    print (des_x)
    print (des_demand)
    print ()
    print ()
    print ("Linear Regression Equation:")
    print ("Y=%fX+%f" % (slope,intercept))
    print ()
    print ("Linear Regression Values:")
    print (reg_demand)
    print ("Seasonal Factors:")
    print (sea_factors)
    print ("Average Seasonal:")
    print (avg_sea)
    print ("Reseasonalized Demand:")
    print (res_demand)

def moving_average(xlsx,dataset):
  moving_average = xlsx.add_worksheet('moving_average')

  # init lists
  lvl = []
  fcast = []
  err = []
  aerr = []
  mse = []
  mad = []
  perr = []

  # our data 
  data = pd.read_csv(dataset)
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])

  # column names
  c_names = ['Period','Demand','Level','Forecast','Error','Absolute Error','Squared Error (MSE)','MAD','% Error', 'MAPE','TS']
  row = 0 
  col = 0
  for name in (c_names):
    moving_average.write (row,col, name)
    col += 1

  # fill out period
  row = 1 
  col = 0
  for x in range (num_demand):
    moving_average.write (row,col, x+1)
    row += 1

  # Demand Data
  row = 1 
  col = 1
  for x in range (num_demand):
    moving_average.write (row,col,demand[x]) 
    row += 1

  # Level
  row = p 
  col = 2
  for x in range ((num_demand-p)+1):
    level = np.average(demand[x:x+p])
    lvl.append(level)
    moving_average.write (row,col,level) 
    row+=1
 
  # Forecast
  row = p+1
  col = 3
  for x in range (len(lvl)-1):
    forecast = lvl[x]
    fcast.append(forecast)
    moving_average.write (row,col,forecast) 
    row+=1
  

  # Error
  row = p+1
  col = 4
  for x in range (len(fcast)):
    error = fcast[x] - demand [x+p]
    err.append(error)
    moving_average.write (row,col,error) 
    row+=1
  
  # Absolute error
  row = p+1
  col = 5
  for x in range (len(err)):
    abserr = np.absolute(err[x]) 
    aerr.append(abserr)
    moving_average.write (row,col,abserr) 
    row+=1

  # Squared root error (MSE)
  row = p+1
  col = 6
  for x in range (len(err)):
    mean_square = np.sum(np.power(err[0:x+1],2))/(x+1)
    mse.append(mean_square)    
    moving_average.write (row,col,mean_square) 
    row+=1

  # Mean absolute Deviation (MAD)
  row = p+1
  col = 7
  for x in range (len(aerr)):
    meanad = np.sum(aerr[0:x+1])/(x+1)
    mad.append(meanad)    
    moving_average.write (row,col,meanad) 
    row+=1

  # % error
  row = p+1
  col = 8
  for x in range (len(err)): 
    percent = 100 * (aerr[x]/demand[x+p])
    perr.append(percent)
    moving_average.write (row,col,percent) 
    row += 1

  # MAPE 
  row = p+1
  col = 9
  for x in range (len(err)): 
    percent = 100 * (aerr[x]/demand[x+p])
    perr.append(percent)
    moving_average.write (row,col,percent) 
    row += 1
    
    
    

def simple_exponential_smoothing(xlsx,dataset):
  s_e = xlsx.add_worksheet('simple_exponential_smoothing')

  # column names
  c_names = ['Period','Demand','Level','Forecast','Error','Absolute Error','Squared Error (MSE)','MAD','% Error', 'MAPE','TS']
  row = 0 
  col = 0
  for name in (c_names):
    s_e.write (row,col, name)
    col += 1


if __name__ == "__main__":
  xlsx = xlsxwriter.Workbook('generated_spreadsheet.xlsx')
  dataset = 'data.csv'
  static_forecast(xlsx,dataset)
  moving_average(xlsx,dataset)
  simple_exponential_smoothing(xlsx,dataset)
  xlsx.close()
