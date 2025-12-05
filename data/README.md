# New-York-Taxi-Routing-Optimization Data Set Up

This guide explains how to set up all datasets used in this project, including how to acquire raw data from NYC Open Data and build the local DuckDB database for analysis.

## Datasets

2016 Yellow Taxi Trip Data: 
Loads from TLC parquet files and from cloud api
Original source - https://data.cityofnewyork.us/Transportation/2016-Yellow-Taxi-Trip-Data/uacg-pexx/about_data 

Automated Traffic Volume Counts: 
Go to this link - https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/about_data 
Select export in the upper right corner and download the csv

Motor Vehicle Collisions - Crashes: 
Go to this link - https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data
Select export in the upper right corner and download the csv

## Loading Data

### Conda Environment

Make a new conda environment
run pip install -r requirements.txt   

### Setting up local persistent duckdb

run build_warehouse.py

python data/duckdb/build_warehouse.py

You will now have a duckdb warehouse file

If needed, we will need to store the duckdb file elsewhere (i.e. S3 bucket) if we are actively making changes that need to build off of one another dynamcially and can't depend on each us running scripts on our end

## Merging Data

run python data/duckdb/merge_and_clean.py 

Now you will have fully built out the Data Warehouse!

## Exploring Data

run python data/duckdb/explore_data.py 

Will also return some basic stats on the warehouse

## Use Cases

Please see scripts housed under data/duckdb/use_cases folder

## EDA

run python data/duckdb/eda_plots.py 


 


duckdb nyc\_traffic\_2016.duckdb
D .read data/spatial\_join.sql
