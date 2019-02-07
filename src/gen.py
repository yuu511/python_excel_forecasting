import xlsxwriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
import shutil

debug = False

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
def static_forecast(xlsx,dataset,dirpath):
  static_forecast = xlsx.add_worksheet('static_foreast')
  data = None

  # init lists
  period      = []
  demand      = []
  des_x       = []
  des_demand  = []
  reg_demand  = []
  sea_factors = []
  avg_sea     = []
  fcast  = []

  # error statistics
  err = []
  aerr = []
  mse = []
  mad = []
  perr = []
  mape = []
  ts = []
  
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
    fcast.append (resea)
    static_forecast.write (row,col,resea)
    row += 1

  # write error column names and calculate error statistics 
  err,aerr,mse,mad,perr,mape,ts = calculate_error(fcast,demand,static_forecast,1,8)
  
  # plot regressed, deseasonalized demand and reseasonalized demand
  plt.plot (period,reg_demand,period,fcast)
  plt.ylabel('demand')
  plt.xlabel('period')
  plt.savefig(os.path.join(dirpath,'static_forecast.png'))
  plt.close()

  # debug print statements
  if (debug):
    print ("period      %r " %  period     )
    print ("demand      %r " %  demand     )
    print ("seasonal_factors %r " %  sea_factors)
    print ("deseasonalized_demand  %r " %  des_demand )
    print ("regressed_demand  %r " %  reg_demand )
    print ("avg_seasonal_factors     %r " %  avg_sea    )
    print ("reseasonalized_demand  %r " %  fcast )

# calculate moving average forecast
def moving_average(xlsx,dataset,dirpath):
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
  data = pd.read_csv(dataset)
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
  plt.plot (period, demand, period[len(period)-len(fcast):len(period)],fcast)
  plt.savefig(os.path.join(dirpath,'moving_average_forecast.png'))
  plt.ylabel('demand')
  plt.xlabel('period')
  plt.close()

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
def simple_exponential_smoothing(xlsx,dataset,dirpath,alpha):
  s_e = xlsx.add_worksheet('simple_exponential_smoothing')

  # our data 
  data = pd.read_csv(dataset)
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
  plt.plot (period, demand, period,fcast)
  plt.ylabel('demand')
  plt.xlabel('period')
  plt.savefig(os.path.join(dirpath,'exponential_smoothing_forecast.png'))
  plt.close()

if __name__ == "__main__":
  src_path = os.path.abspath(__file__)
  src_dir =  os.path.dirname(src_path)
  dirname= 'graphs+excel_directory'
  dataset = os.path.join(src_dir , 'data.csv')
  if (len(sys.argv) >= 2):
    dataset = sys.argv[1]
    if not (os.path.exists(dataset)):
      print (" Dataset : %s does not exist!" % dataset)
      sys.exit(1)
    dataset = os.path.abspath(dataset)
  if (len(sys.argv) >= 3):
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

  # make graph directory to store all graph pictures in
  graphpath = os.path.abspath(os.path.join(dirname,'graphs'))
  os.mkdir(graphpath)
  dataset   = os.path.abspath(dataset)
  dirname   = os.path.abspath(dirname)
  xlsx = xlsxwriter.Workbook(os.path.join(dirname,'generated_spreadsheet.xlsx'))
  static_forecast(xlsx,dataset,graphpath)
  moving_average(xlsx,dataset,graphpath)
  alpha = 0.1
  simple_exponential_smoothing(xlsx,dataset,graphpath,alpha)
  xlsx.close()
