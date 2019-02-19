import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import shutil
import copy
import math

debug = False

# simple class object to return original demand , forecast , and error statistics
class forecast_results:
  def __init__ (self,demand,forecast,err,aerr,mse,mad,perr,mape,ts):
    self.demand = demand
    self.forecast = forecast
    self.err = err
    self.aerr = aerr
    self.mse = mse
    self.mad = mad
    self.perr = perr
    self.mape = mape
    self.ts = ts
  def get_demand(self):
   return self.demand
  def get_forecast(self):
   return self.forecast
  def get_error(self):
   return self.err
  def get_absolute_error(self):
   return self.aerr
  def get_MSE(self):
   return self.MSE
  def get_MAD(self):
   return self.mad
  def get_percent_error(self):
   return self.perr
  def get_MAPE(self):
   return self.mape
  def get_tracking_signal(self):
   return self.ts

# plot demand vs forecast
def graph_fcast_demand(fperiod,forecast,period,demand,name,path):
  plt.plot (fperiod,forecast,period,demand)
  plt.ylabel('demand')
  plt.xlabel('period')
  plt.savefig(os.path.join(path,name))
  plt.close()
  
# generate error statistics (Error,ABS error, MSE , MAD, MAPE and TS)
# based on forecast,error and demand, write on specified excel worksheet (w) beginning on (row,col)
# NOTE: demand must equal the size of forecast for the calculation to be correct (Do array slicing BEFORE calling the function)
# (indexes of demand and fcast will be compared directly)
def calculate_error(fcast,demand,w,row,col):
  o_row = row
  o_col = col

  err = []
  aerr = []
  mse = []
  mad = []
  perr = []
  mape = []
  ts = []

  # write error name column names
  error_names = ['Error','Absolute Error','Squared Error (MSE)','MAD','% Error', 'MAPE','TS']
  row = 0
  for name in (error_names):
    w.write (row,col, name)
    col += 1

  # Error
  row = o_row
  col = o_col
  for x in range (len(fcast)):
    error = fcast[x] - demand [x]
    err.append(error)
    w.write (row,col,error) 
    row+=1

  # Absolute error
  row = o_row
  col+=1
  for x in range (len(err)):
    abserr = np.absolute(err[x]) 
    aerr.append(abserr)
    w.write (row,col,abserr) 
    row+=1
    
  
  # Squared root error (MSE)
  row = o_row
  col +=1
  for x in range (len(err)):
    mean_square = np.sum(np.power(err[0:x+1],2))/(x+1)
    mse.append(mean_square)    
    w.write (row,col,mean_square) 
    row+=1


  # Mean absolute Deviation (MAD)
  row = o_row
  col +=1
  for x in range (len(aerr)):
    meanad = np.sum(aerr[0:x+1])/(x+1)
    mad.append(meanad)    
    w.write (row,col,meanad) 
    row+=1

  # % error
  row = o_row
  col +=1
  for x in range (len(aerr)): 
    percent = 100 * (aerr[x]/demand[x])
    perr.append(percent)
    w.write (row,col,percent) 
    row += 1

  # MAPE 
  row = o_row
  col +=1
  for x in range (len(perr)): 
    mpe = np.sum(perr[0:x+1])/(x+1)
    mape.append(mpe)    
    w.write (row,col,mpe) 
    row+=1
     
  # tracking signal
  row = o_row
  col +=1
  for x in range (len(err)): 
    tracking = np.sum(err[0:x+1])/mad[x]
    ts.append(tracking)    
    w.write (row,col,tracking) 
    row+=1

  return err,aerr,mse,mad,perr,mape,ts
   
# calcualte static forecast
def static_forecast(xlsx,data,dirpath):
  static_forecast = xlsx.add_worksheet('static_foreast')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])

  # init lists
  des_x       = []
  des_demand  = []
  reg_demand  = []
  sea_factors = []
  fcast       = []
  fcasted = []
  avg_sea     = [0] * p
  occurences_s= copy.deepcopy(avg_sea)
  num_predicted = p

  # error statistics
  err = []
  aerr = []
  mse = []
  mad = []
  perr = []
  mape = []
  ts = []
  
  
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
      upper = (lower + p)
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

  slope,intercept, r_value, p_value, std_err = stats.linregress(des_x,des_demand)

  row = 1
  col = 4
  for x in range (num_demand):
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
  for x in range (len(sea_factors)):
    mod = x%p
    avg_sea[mod]  += sea_factors[x] 
    occurences_s[mod] += 1 

  for x in range (len (avg_sea)):
    avg_sea[x] = avg_sea[x] / occurences_s[x]
    static_forecast.write (row,col,avg_sea[x])
    row += 1
  
  # reseasonalize demand
  row = 1
  col = 7
  for x in range (num_demand):
    resea =  avg_sea[x%p] * reg_demand[x]
    fcast.append (resea)
    static_forecast.write (row,col,resea)
    row += 1

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand,static_forecast,1,8)
  
  # forecast predict ( up to demand length + p in this case ) 
  row = (num_demand + 1)
  col = 7
  for x in range (num_demand,num_demand+num_predicted):
    period = np.concatenate([period,[x+1]])
    predict = avg_sea[x%p] * ((slope * (x+1))+intercept)
    fcast.append(predict)
    fcasted.append(predict)
    static_forecast.write (row,col,predict) 
    row +=1
  
  # plot regressed, deseasonalized demand and reseasonalized demand
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'static_forecast',dirpath)

  # debug print statements
  if (debug):
    print ("period      %r " %  period     )
    print ("demand      %r " %  demand     )
    print ("seasonal_factors %r " %  sea_factors)
    print ("deseasonalized_demand  %r " %  des_demand )
    print ("regressed_demand  %r " %  reg_demand )
    print ("avg_seasonal_factors     %r " %  avg_sea    )
    print ("reseasonalized_demand  %r " %  fcast )

  # return linear regression of deseasonalized data and average seasonal factors for winter regression
  f = forecast_results(demand,fcast,err,aerr,mse,mad,perr,mape,ts)
  return f,slope,intercept,avg_sea

# calculate moving average forecast
def moving_average(xlsx,data,dirpath):
  moving_average = xlsx.add_worksheet('moving_average')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])
  num_predicted = p

  # init lists
  lvl = []
  fcast = []
  err = []
  aerr = []
  mse = []
  mad = []
  perr = []
  mape = []
  ts = []
  fcasted = []


  # column names
  c_names = ['Period','Demand','Level','Forecast']
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

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand[p:len(demand)],moving_average,p+1,4)

  # forecast predict ( up to demand length + p in this case ) 
  row = (len(fcast) + 1 + p)
  col = 3
  for x in range (len(fcast),len(fcast)+num_predicted):
    # period = np.concatenate([period,[x+1]])
    predict = lvl[len(lvl)-1]
    fcast.append(predict)
    fcasted.append(predict)
    moving_average.write (row,col,predict) 
    row +=1

  # plot moving average forecast
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'moving_average_forecast.png',dirpath)

  # # plot the forecast
  # graph_fcast_demand(period, demand, period[len(period)-len(fcast):len(period)],fcast,'moving_average_forecast',dirpath)

  # debug
  if (debug):
    print("lvl  \n %r" % lvl  )
    print("fcast\n %r" % fcast)
    print("err  \n %r" % err  )
    print("aerr \n %r" % aerr )
    print("mse  \n %r" % mse  )
    print("mad  \n %r" % mad  )
    print("perr \n %r" % perr )
    print("mape \n %r" % mape )
    print("ts   \n %r" % ts   )

  f = forecast_results(demand,fcast,err,aerr,mse,mad,perr,mape,ts)
  return f

# calculate forecast with simple exponential smoothing
def simple_exponential_smoothing(xlsx,data,dirpath,alpha):
  s_e = xlsx.add_worksheet('simple_exponential_smoothing')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])

  # init lists
  lzero = None
  lvl   = []
  fcast = []
  fcasted = []
  err   = []
  aerr  = []
  mse   = []
  mad   = []
  perr  = []
  mape  = []
  ts    = []

  # column names
  c_names = ['Period','Demand','Level','Forecast']
  row = 0 
  col = 0
  for name in (c_names):
    s_e.write (row,col, name)
    col += 1

  # calculate Level L0
  row = 1
  col = 0
  s_e.write (row,col,0)
  lzero = np.average(demand)
  s_e.write (row,col+2,lzero)
  
 
  # fill out period
  row = 2 
  col = 0
  for x in range (num_demand):
    s_e.write (row,col, x+1)
    row += 1

  # Demand Data
  row = 2 
  col = 1
  for x in range (num_demand):
    s_e.write (row,col,demand[x]) 
    row += 1

  # Level
  row = 2 
  col = 2
  lvl.append(lzero)
  for x in range (num_demand):
    level = (alpha * demand[x]) + ((1 - alpha) * lvl[x]) 
    lvl.append(level)
    s_e.write (row,col,level) 
    row+=1
 
  # Forecast
  row = 2
  col = 3
  for x in range (len(lvl)-1):
    forecast = lvl[x]
    fcast.append(forecast)
    s_e.write (row,col,forecast) 
    row+=1
 

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand,s_e,2,4)

  # forecast predict ( up to demand length + p in this case ) 
  row = (len(fcast) + 2 )
  col = 3
  for x in range (len(fcast),len(fcast)+p):
    period = np.concatenate([period,[x+1]])
    predict = lvl[len(lvl)-1]
    fcast.append(predict)
    fcasted.append(predict)
    s_e.write (row,col,predict) 
    row +=1

  # plot simple exp smoothing
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'simple_exp_smoothing.png',dirpath)

  # debug
  if (debug):
    print("lvl  \n %r" % lvl  )
    print("fcast\n %r" % fcast)
    print("err  \n %r" % err  )
    print("aerr \n %r" % aerr )
    print("mse  \n %r" % mse  )
    print("mad  \n %r" % mad  )
    print("perr \n %r" % perr )
    print("mape \n %r" % mape )
    print("ts   \n %r" % ts   )

  f = forecast_results(demand,fcast,err,aerr,mse,mad,perr,mape,ts)
  return f

# calculate forecast with holt trend smoothing
def holt_trend_corrected_exponential_smoothing(xlsx,data,dirpath,alpha,beta):
  ht = xlsx.add_worksheet('holt_trend_exp_smoothing')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])
  num_predicted = p

  # init lists
  lzero = None
  tzero = None
  lvl   = []
  tnd   = []
  fcast = []
  fcasted = []
  err   = []
  aerr  = []
  mse   = []
  mad   = []
  perr  = []
  mape  = []
  ts    = []
  
  # column names
  c_names = ['Period','Demand','Level','Trend','Forecast']
  row = 0 
  col = 0
  for name in (c_names):
    ht.write (row,col, name)
    col += 1

  slope, intercept, r_value, p_value, std_err = stats.linregress(period,demand)

  # calculate Level L0
  row = 1
  col = 0
  ht.write (row,col,0)
  lzero = intercept
  ht.write (row,col+2,lzero)
  
  # calculate Trend T0
  row = 1
  col = 3
  tzero = slope
  ht.write (row,col,tzero)
  
  # fill out period
  row = 2 
  col = 0
  for x in range (num_demand):
    ht.write (row,col, x+1)
    row += 1

  # Demand Data
  row = 2 
  col = 1
  for x in range (num_demand):
    ht.write (row,col,demand[x]) 
    row += 1

  # Level + trend
  row = 2 
  col = 2
  lvl.append(lzero)
  tnd.append(tzero)
  for x in range (num_demand):
    level = (alpha * demand[x]) + ((1 - alpha) * (lvl[x]+tnd[x])) 
    lvl.append(level)
    trend = (beta * (lvl[x+1] - lvl [x])) + ((1 - beta) * tnd[x]) 
    tnd.append(trend)
    ht.write (row,col,level) 
    ht.write (row,col+1,trend) 
    row+=1
 
  # Forecast
  row = 2
  col = 4
  for x in range (len(lvl)-1):
    forecast = lvl[x] + tnd[x]
    fcast.append(forecast)
    ht.write (row,col,forecast) 
    row+=1
 
  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand,ht,2,5)

  # forecast predict ( up to demand length + p in this case ) 
  row = num_demand+2
  col = 4
  for x in range (num_demand,num_demand+num_predicted):
    period = np.concatenate([period,[x+1]])
    print (x-num_demand+1)
    predict = (lvl[len(lvl)-1] + ((x-num_demand+1) * tnd[len(tnd)-1]))
    fcast.append(predict)
    fcasted.append(predict)
    ht.write (row,col,predict) 
    row +=1

  # plot holt smoothing
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'holt_trend_forecast.png',dirpath)

  # debug
  if (debug):
    print("lvl  \n %r" % lvl  )
    print("tnd  \n %r" % tnd  )
    print("fcast\n %r" % fcast)
    print("err  \n %r" % err  )
    print("aerr \n %r" % aerr )
    print("mse  \n %r" % mse  )
    print("mad  \n %r" % mad  )
    print("perr \n %r" % perr )
    print("mape \n %r" % mape )
    print("ts   \n %r" % ts   )

  f = forecast_results(demand,fcast,err,aerr,mse,mad,perr,mape,ts)
  return f

# calculate forecast with simple exponential smoothing
def winter_trend_seasonality_forecast(xlsx,data,dirpath,alpha,beta,gamma,slope,intercept,avg_sea):
  wt = xlsx.add_worksheet('winter_trendseason')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])
  num_predicted = p

  # init lists
  lzero = None
  tzero = None
  sea   = []
  lvl   = []
  tnd   = []
  fcast = []
  err   = []
  aerr  = []
  mse   = []
  mad   = []
  perr  = []
  mape  = []
  ts    = []
  fcasted = []
   
  # column names
  c_names = ['Period','Demand','Level','Trend','Seasonal Factor','Forecast']
  row = 0 
  col = 0
  for name in (c_names):
    wt.write (row,col, name)
    col += 1

  # calculate Level L0 (Linregress of Deseasonalized data is calculated in static forecasting, transferred here)
  row = 1
  col = 0
  wt.write (row,col,0)
  lzero = intercept
  wt.write (row,col+2,lzero)
  
  # calculate Trend T0 (Linregress of Deseasonalized data is calculated in static forecasting, transferred here)
  row = 1
  col = 3
  tzero = slope
  wt.write (row,col,tzero)
  
  # fill out period
  row = 2 
  col = 0
  for x in range (num_demand+num_predicted):
    wt.write (row,col, x+1)
    row += 1

  # Demand Data
  row = 2 
  col = 1
  for x in range (num_demand):
    wt.write (row,col,demand[x]) 
    row += 1

  for x in range (len(avg_sea)):
    sea.append(avg_sea[x])  

  # Level + trend + seasonal factor (up to number of data we have)
  row = 2 
  col = 2
  lvl.append(lzero)
  tnd.append(tzero)
  for x in range (num_demand):
    level = ((gamma* (demand[x]/sea[x]))) + ((1 - gamma) * (lvl[x]+tnd[x])) 
    lvl.append (level)
    wt.write (row,col,level) 
    trend = ((alpha * (lvl[x+1] - lvl [x])) + ((1- alpha)*tnd[x])) 
    tnd.append (trend)
    wt.write (row,col+1,trend) 
    seasonalf = ((alpha * (demand[x]/lvl[x+1])) + ((1 - alpha)*sea[x]))
    wt.write (row,col+2,seasonalf) 
    sea.append (seasonalf)
    row += 1

  # generate forecast (up to demand data length)
  row = 2
  col = 5
  for x in range (num_demand):
    forecast =  (lvl[x] + tnd[x]) * sea[x]
    fcast.append(forecast)
    wt.write (row,col,forecast)
    row += 1
  

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand,wt,2,6)

  # forecast predict ( up to demand length + p in this case ) 
  row = num_demand+2
  col = 4
  for x in range (num_demand,num_demand+num_predicted):
    period = np.concatenate([period,[x+1]])
    predict = (lvl [num_demand-1] + ((x-num_demand) * tnd[num_demand-1]))* sea [x] 
    fcast.append(predict)
    fcasted.append(predict)
    wt.write (row,col,sea[x]) 
    wt.write (row,col+1,predict) 
    row +=1

  # plot winter smoothing
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'winter_trend_season_forecast.png',dirpath)

  # debug
  if (debug):
    print("lvl  \n %r" % lvl  )
    print("tnd  \n %r" % tnd  )
    print("fcast\n %r" % fcast)
    print("err  \n %r" % err  )
    print("aerr \n %r" % aerr )
    print("mse  \n %r" % mse  )
    print("mad  \n %r" % mad  )
    print("perr \n %r" % perr )
    print("mape \n %r" % mape )
    print("ts   \n %r" % ts   )

  f = forecast_results(demand,fcast,err,aerr,mse,mad,perr,mape,ts)
  return f

