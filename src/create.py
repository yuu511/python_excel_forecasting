import xlsxwriter

workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet('param')
# automate writing of column names
c_names = ['Year','Quarter','Period','Demand','De-Seasonalized Demand','Regressed,Deseasonalized Demand', 'Seasonal Factors', 'Average Seasonal Factors', 'Reintroduce Seasonality']
row = 0
col = 0

for name in (c_names):
  print (name)
  print (col)
  worksheet.write (row,col, name)
  row +=1
  col+=1

workbook.close()
