# EDA Subfolder

This folder contains all of the raw data and the scripts for the interactive visualizations, as well as the NetKDE and hierarchical clustering for carpooling.

- cache - osmnx road graph cache (not relevant)
- geo - geospatial files, specifically the boundary files for nyc boroughs, which were used to filter for relevant trips
- maps - output of interactive visualizations using Folium, in the form of html files, which can be opened and interacted with in browser
- netkde_out - contains node appearence frequency counts and node pairing (trip origin-destination pairs) which represent the preprocessing step of the netKDE method.
- netkde.ipynb - contains the code for the netKDE method, including filtering of trips within manhattan, assignment of origin destinations to osmnx intersection nodes, calculation of occurence counts of unique node pairs, estimation of congestion per road intersection.
- taxi_eda.ipynb - contains the scripts for visualization of taxi and collision datasets, as well as hierarchical clustering.
