# Carpooling Optimization & Shared Ride Prediction  

**Authors:** Tiger & Anusree  

## Overview  

This project explores shared ride optimization in New York City taxis to reduce congestion. The pipeline identifies opportunities where multiple trips can be pooled together based on proximity and timing, and evaluates the potential congestion reduction from these shared rides.  

Two different ride-share matching algorithms were implemented and compared:  

1. **K-means / Clustering** (Anusree & Tiger)  
   - Groups trips based on geographic proximity using clustering techniques.  
2. **Distance-Time Threshold Matching** (Anusree)  
   - Pairs trips that are close in both pickup and dropoff locations, and have small differences in pickup time, respecting the maximum taxi capacity.  

## Objectives  

- Determine where shared rides could reduce congestion.  
- Compare different ride-share matching algorithms.  
- Estimate total congestion reduction achievable from shared rides.  
- Generate actionable outputs:  
  - Carpooling recommendations  
  - Congestion impact analysis of shared rides  

## Distance-Time Threshold Matching Algorithm  

The distance-time threshold method works as follows:  

1. Load a subset of taxi trips from `taxi_clean`.  
2. Iterate through trips sorted by pickup time.  
3. For each trip, check forward trips within a specified **time window**:  
   - Compare pickup locations within a **pickup radius** (e.g., 200m, 400m, 600m).  
   - Compare dropoff locations within a **dropoff radius**.  
   - Ensure the combined passenger count does not exceed the taxi's maximum capacity.  
4. Pair trips that satisfy all thresholds.  
5. Record metrics for each matched pair (distance, time difference, total passengers).  
6. Estimate congestion reduction: each matched pair reduces the total number of cars by 1.  

### Parameters  

- **Max Capacity:** 4 passengers per taxi  
- **Pickup Radii:** 0.2 km, 0.4 km, 0.6 km  
- **Dropoff Radii:** 0.2 km, 0.4 km, 0.6 km  
- **Time Windows:** 3, 5, 7 minutes  

### Outputs  

- Matched trip pairs with details of pickup/dropoff distance, time difference, and passenger counts.  
- Estimated congestion reduction statistics:  
  - Total trips  
  - Number of shared pairs  
  - Reduced trips  
  - Percent reduction in total trips  

## Running the Pipeline  

python distance_time_threshold_carpooling.py

## K Means/Clustering 

Please see the Agglomerative Clustering to Identify Carpooling in the eda jupyter notebook

