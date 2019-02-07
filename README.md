Generates a static, moving average, and simple exponential smoothing excel spreadsheet.

REQUIREMENTS:
-  Python3
-  pip for Python3
-  packages in requirements.txt

OPTIONAL:
-  tk (or jupyter notebook) : Displays matplotlib graphs 

BUILD:
-  python3 -m pip install -r requirements.txt --user

RUN:
Three different ways of running the program.

No arguments:
- %  python3 gen.py : Runs the program with the default src/data.csv  as the dataset. Creates a directory (graphs+excel_directory) with an excel file generated_spreadsheet.xlsx
and a subdirectory graphs with forecast diagrams

1 Argument:
- %  python3 gen.py [DATASET] : Runs the program with [DATASET]  as the dataset. Creates a directory (graphs+excel_directory) with an excel file generated_spreadsheet.xlsx
and a subdirectory graphs with forecast diagrams
- example: %python3 gen.py data2.csv

2 Arguments:

- %  python3 gen.py [DATASET] [FILEPATH] : Runs the program with [DATASET]  as the dataset. Creates a directory [FILEPATH] with an excel file generated_spreadsheet.xlsx
and a subdirectory graphs with forecast diagrams
- example: %python3 gen.py /home/foo/bar 

DATA:
Data is formatted in a CSV file: data.csv.
PERIOD / DEMAND / PERIODICITY

- Where period is any ascending numerical series (1,2,3,4,5...)

- Demand is any amount of numbers

- Periodicity: Single number (which will denote the seasonality of the data)

( look at data.csv for an example. )
