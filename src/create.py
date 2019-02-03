import xlsxwriter
import sys
import numpy as np
import pandas as pd

def main():
  workbook = xlsxwriter.Workbook('hello.xlsx')
  worksheet = workbook.add_worksheet('param')
  
  # our data 
  data = pd.read_csv('data.csv')
  period = data['period'].values
  demand = data['demand'].values 
  periodicity = data['periodicity'].values
  num_periods = len(period)
  p = int(periodicity[0])
  
  # column names
  c_names = ['Year','Period','Demand','De-Seasonalized Demand','Regressed,Deseasonalized Demand', 'Seasonal Factors', 'Average Seasonal Factors', 'Reintroduce Seasonality']
  row = 0 
  col = 0
  for name in (c_names):
    worksheet.write (row,col, name)
    col += 1
  
  # fill out year and period columns
  row = 1 
  col = 0
  for x in range (num_periods):
    mod = x%p
    quarter = p / 4
    if (mod == 0):
      worksheet.write (row,col, (x/p)+1)
    worksheet.write  (row,col+1, x+1)
    row += 1
  
  # Demand Data
  row = 1 
  col = 2
  for x in range (num_periods):
    worksheet.write (row,col,demand[x]) 
    row += 1
  
  # Demand data, deseasonalized
  if (p % 2 == 0): 
    min_val= int((p/2) + 1)
    max_val= int(num_periods - (p/2) + 1)
    row = min_val
    col = 3
    for x in range (min_val,max_val):
      # extra -1 for array indices
      lower = int(x-(p/2))-1
      upper = int(x+(p/2))-1
      des=(demand[lower] + demand[upper]+(2 * np.sum(demand[lower+1:upper])))/(2*p)
      worksheet.write (row,col,des) 
      row += 1
  else:
    min_val = int (((p-1)/2)+1)
    max_val = int (num_periods - ((p-1)/2)+1) 
    row = min_val
    col = 3
    for x in range (min_val,max_val):
      lower = int (x-((p-1)/2)) 
      upper = int (x+((p-1)/2)) 
      des = (np.sum(demand[lower-1:upper]))/p
      worksheet.write (row,col,des)
      row += 1
  
  # Demand data, deseasonalized and regressed
  # X = np.stack ((x,demand),axis=0)
    
  
  workbook.close()

if __name__ == "__main__":
  main() 
