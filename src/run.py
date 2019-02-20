import forecast
import cycle
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

  # read and check validity of dataset before doing anything
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
  beta  = 0.2
  gamma = 0.1
 
  print ("alpha:")
  print (alpha)
  print ("beta:")
  print (beta)
  print ("gamma:")
  print (gamma)

  # make the forecasts
  # return slope,intercept of regression and seasonal factors of deseasonalized demand for winter forecasting
  static_fcast, slope, intercept, avg_sea = forecast.static_forecast(xlsx,data,graphpath)
  moving_average = forecast.moving_average(xlsx,data,graphpath)
  simple_exponential_smoothing = forecast.simple_exponential_smoothing(xlsx,data,graphpath,alpha)
  holt = forecast.holt_trend_corrected_exponential_smoothing(xlsx,data,graphpath,alpha,beta)
  winter = forecast.winter_trend_seasonality_forecast(xlsx,data,graphpath,alpha,beta,gamma,slope,intercept,avg_sea)
  QL = cycle.cycle_inventory(xlsx,winter.get_prediction(),7500,0.1,1000) 
  xlsx.close()
