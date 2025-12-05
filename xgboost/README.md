# Travel Time Estimation with Collisions + Traffic Context


## Directory Listing

```
├── extract_osrm_feats.py
├── fit-xgboost-with-crash-traffic-data.ipynb
├── process_trips.sh
├── README.md
```
* Please see the jupyter notebook `fit-xgboost-with-crash-traffic-data.ipynb` for the feature extraction, training, and evaluation code
* Datasets can be downloaded from [sharepoint](https://gtvault-my.sharepoint.com/:u:/g/personal/kworathur3_gatech_edu/IQCdhMWCgbEFToE8FQEdoE9iARjjsT-Wzm06bi2LPrcTRhw?e=l1r9mj) and should be stored in a folder called `input` in the project root.
* If you would like to compute travel times using an API for a large number of trips, I recommend **self hosting** an OSRM server. Here's an excellent guide for how to deploy your own OSRM server on AWS [blog link](https://blog.afi.io/blog/hosting-the-osrm-api-on-amazon-ec2-running-osrm-backend-as-a-web-service/)
* Other contributions: found a bug in [py-osrm-client](https://github.com/tomrss/py-osrm-client) code and submitted a fix that approved by the project maintainers
