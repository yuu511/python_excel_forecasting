import math
import numpy as np

def cycle_inventory(xlsx,demand,S,h,C):
  cycle_inventory = xlsx.add_worksheet('Find_EOQ_cycle_inventory')
  D = np.sum(demand)
  ret = math.sqrt((2*D*S)/(h*C))  
  c_names = ['Annual demand (Forecasted,cumulative)','Shipment cost','Holding cost ', 'Material Cost ']
  row = 0
  col = 0
  for name in (c_names):
    cycle_inventory.write (row,col,name)
    col += 1

  row = 1
  col = 0
  cycle_inventory.write(row,col,D)
  cycle_inventory.write(row,col+1,S)
  cycle_inventory.write(row,col+2,h)
  cycle_inventory.write(row,col+3,C)
  row = 2  
  col = 0
  cycle_inventory.write(row,col,"OPTIMAL LOT SIZE")
  cycle_inventory.write(row+1,col,ret)
  row = 2  
  col = 1
  cycle_inventory.write(row,col,"Cycle Inventory")
  cycle_inventory.write(row+1,col,ret/2)
  row = 4 
  col = 0
  cycle_inventory.write(row,col,"Number of Orders Per Year:")
  cycle_inventory.write(row+1,col,D/ret)
  cycle_inventory.write(row,col+1,"Replenishment time:")
  cycle_inventory.write(row+1,col+1,365/(D/ret))
  row = 6 
  col = 0
  cycle_inventory.write(row,col,"Average Flow time:")
  cycle_inventory.write(row+1,col,(ret/(2*D))*365)
  return ret
