## Traffic and Road Closure Data Analysis


This folder includes SQL scripts to read New York traffic data from CSV files into a table, and a python script to transform data annotated with Node IDs to Segment IDs to support joins between datasets.


| File              | Description                                                                                                     |
|-------------------|-----------------------------------------------------------------------------------------------------------------|
| `create_tables.sql` | Loads the automated traffic counts, road closure (blocks) and road closure (intersections) datasets into DuckDB |
| `node_to_segments.py`     |     Converts the Node ID column in a CSV file to Segment ID column, possibly generating more than one row for each row in the input file.          

### Getting Started


1. Install duckdb

On MacOS, run the commands below in a terminal 

```bash
$ brew install duckdb
$ duckdb -init create_tables.sql
```

2. Create a virtual environment and install dependencies

```bash
python -m venv <your_env_name>
activate <your_env_name>
cd traffic_and_closures_data
pip install -r requirements.txt
```

3. Download the "Automated Traffic Counts" and "Road Closures" by Intersection datasets from NYC Open data

```bash
$ ./download_data.sh
```

