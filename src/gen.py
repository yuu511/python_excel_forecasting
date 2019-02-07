import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import shutil
import copy

debug = False

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
  avg_sea     = [0] * p
  occurences_s= copy.deepcopy(avg_sea)

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
  slope, intercept, r_value, p_value, std_err = stats.linregress(des_x,des_demand)
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
  
  # plot regressed, deseasonalized demand and reseasonalized demand
  graph_fcast_demand(period,demand,period,fcast,'static_forecast',dirpath)

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
  return slope,intercept,avg_sea

# calculate moving average forecast
def moving_average(xlsx,data,dirpath):
  moving_average = xlsx.add_worksheet('moving_average')

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

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])

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

  # plot the forecast
  graph_fcast_demand(period, demand, period[len(period)-len(fcast):len(period)],fcast,'moving_average_forecast',dirpath)

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

  # plot simple exponential smoothing graph
  graph_fcast_demand(period,demand,period,fcast,'exponential_smoothing_forecast.png',dirpath)

# calculate forecast with simple exponential smoothing
def holt_trend_corrected_exponential_smoothing(xlsx,data,dirpath,alpha,beta):
  ht = xlsx.add_worksheet('holt_trend_exp_smoothing')

  # our data 
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_demand = len(demand)
  num_period = len(period)
  p = int(periodicity[0])

  # init lists
  lzero = None
  tzero = None
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
  ht.write (row,col,slope)
  
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

  # plot holt smoothing
  graph_fcast_demand(period,demand,period,fcast,'holt_trend_correct_forecast.png',dirpath)

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
  num_fcast = p
   
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
  for x in range (num_demand+num_fcast):
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
  
  # forecast predict ( up to demand length + p in this case ) 
  row = num_demand+2
  col = 4
  for x in range (num_demand,num_demand+num_fcast):
    period = np.concatenate([period,[x+1]])
    predict = (lvl [num_demand-1] + ((x-num_demand) * tnd[num_demand-1]))* sea [x] 
    fcast.append(predict)
    fcasted.append(predict)
    wt.write (row,col,sea[x]) 
    wt.write (row,col+1,predict) 
    row +=1

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast[0:num_demand],demand,wt,2,6)

  # plot winter smoothing
  graph_fcast_demand(period[0:num_demand],demand,period,fcast,'winter_trend_season_forecast.png',dirpath)

  print (period)
  print (fcast)
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

if __name__ == "__main__":
  src_path = os.path.abspath(__file__)
  src_dir =  os.path.dirname(src_path)
  root_dir = os.path.normpath(os.path.join(src_dir,os.pardir))
  data_dir = os.path.join(root_dir,'data')
  if not (os.path.exists(data_dir)):
    print("Data directory %s does not exist!" % data_dir)
    sys.exit(1)
  
  dirname= 'graphs+excel_directory'
  dataset = os.path.join(data_dir , 'data.csv')
  arglen = len(sys.argv)
  if (arglen >= 2):
    dataset = sys.argv[1]
    if not (os.path.exists(dataset)):
      print (" Dataset : %s does not exist!" % dataset)
      sys.exit(1)
    dataset = os.path.abspath(dataset)
  if (arglen >= 3):
    dirname = sys.argv[2]
  # make directory for all files to be stored in(excel file, and graphs)
  if (os.path.exists(dirname)):
    while (1):
      print ("The directory you are trying to save your files to :\n[ %s ]\nalready exists. delete it? y/n" % dirname)
      try:
        line = sys.stdin.readline()
      except KeyboardInterrupt:
         break
      if not line:
        break
      if (line.rstrip() is 'y'):
        print ("Deleting...")
        try:
          shutil.rmtree(dirname)
          os.mkdir(dirname)
        except (OSError,IOError) as e:
          print ("Error in making directory!") 
          sys.exit(1)
        print ("Deletion sucess: folder %s overwritten!" % dirname)
        break;
      elif (line.rstrip() is 'n'):
        print ("Exiting...")
        sys.exit()
  else:
    try:
      os.mkdir(dirname)
    except OSError as e:
      print ("Error in making directory!") 
      sys.exit(1)

  # read andcheck validity of dataset before doing anything
  dataset   = os.path.abspath(dataset)
  data = pd.read_csv(dataset)
  if not ('period' in data):
    print ("no period in csv")
    sys.exit(1)
  if not ('demand' in data):
    print ("no demand data")
    sys.exit(1)
  if not ('periodicity' in data):
    print ("no periodicity data")
    sys.exit(1)
  periodicity = data['periodicity'].values
  if not (isinstance(periodicity[0],float)): 
    print ("no periodicity data")
    sys.exit(1)

  # make graph directory to store all graph pictures in
  graphpath = os.path.abspath(os.path.join(dirname,'graphs'))
  os.mkdir(graphpath)
  dirname   = os.path.abspath(dirname)
  xlsx = xlsxwriter.Workbook(os.path.join(dirname,'generated_spreadsheet.xlsx'))
  alpha = 0.1
  beta = 0.2
  gamma = 0.05

  # make the forecasts
  # return slope,intercept of regression and seasonal factors of deseasonalized demand for winter forecasting
  slope, intercept, avg_sea = static_forecast(xlsx,data,graphpath)
  moving_average(xlsx,data,graphpath)
  simple_exponential_smoothing(xlsx,data,graphpath,alpha)
  holt_trend_corrected_exponential_smoothing(xlsx,data,graphpath,alpha,beta)
  winter_trend_seasonality_forecast(xlsx,data,graphpath,alpha,beta,gamma,slope,intercept,avg_sea)
  xlsx.close()
