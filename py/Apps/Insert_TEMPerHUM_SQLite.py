#Script was developed JBS to Analyze Humidity Data Collected via csv file at routine interval and dump data into SQLite database

import datetime
import sqlite3
from sqlite3.dbapi2 import Time, Timestamp
import pandas as pd


print("Beginning SQLite Database Addition for TEMPerHUM Data Refresh:")
StartTime = datetime.datetime.now()
print("Start Time: ", StartTime)

conn = sqlite3.connect('PATH TO SQLITE DB FILE')
c = conn.cursor()

# load the data into a Pandas DataFrame
TEMPerHUM = pd.read_csv('PATH TO CSV FILE (with TEMPerHUM data)')

# write the data to a sqlite table
TEMPerHUM.to_sql('TEMPerHUM_Data', conn, if_exists='append', index = False)

EndTime = datetime.datetime.now()
print("End Time: ", EndTime)

print("Overview: ", StartTime, EndTime)

ExecutionTime = (EndTime-StartTime)
print("Script Execution Time: ", ExecutionTime)

print("Complete! :)")