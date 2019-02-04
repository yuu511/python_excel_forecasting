REQUIREMENTS:
-  Python3
-  pip 
-  packages in requirements.txt

BUILD:
-  pip install -r requirements.txt

RUN:
-  python gen.py

DATA:
Data is formatted in a CSV file: data.csv.
PERIOD / DEMAND / PERIODICITY

Where period is any ascending numerical series (1,2,3,4,5...)
Demand is any amount of numbers
Periodicity: Single number (which will denote the seasonality of the data)

Look at data.csv for an example.
