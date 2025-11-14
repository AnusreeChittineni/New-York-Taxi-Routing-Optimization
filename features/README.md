# Edge Feature Extraction from NYC Taxi Trip Data

To extract graph features from data stored in duckdb:

```bash
$ python build_graph_fets.py --duckdb ../nyc_traffic_2016.duckdb --trips_table taxi_data --max_rows 100000 
```

To create a heatmap of a specific graph features, run the script below:

```bash
$ python viz_graph_fets.py --data_dir ./nyc_gnn_data --metric congestion_factor --output_prefix nyc_graph_report
```
