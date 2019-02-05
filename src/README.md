Generates a static, moving average, and simple exponential smoothing excel spreadsheet in the same directory the program is run.
Run gen.py in the src folder.

REQUIREMENTS:
-  Python3
-  pip for Python3
-  tk (or jupyter notebook)
-  packages in requirements.txt

BUILD:
<<<<<<< HEAD
-  python3 -m pip install --user -r requirements.txt
=======
-  pip install -r requirements.txt --user
>>>>>>> d38128ad0615aedb6780d2451b02e354448a7d36

RUN:
-  python gen.py

DATA:
Data is formatted in a CSV file: data.csv.
PERIOD / DEMAND / PERIODICITY

- Where period is any ascending numerical series (1,2,3,4,5...)

- Demand is any amount of numbers

- Periodicity: Single number (which will denote the seasonality of the data)

( look at data.csv for an example. )
