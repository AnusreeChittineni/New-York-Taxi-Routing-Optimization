# New York Taxi Routing Optimization Data Setup

This guide explains how to set up all datasets used in this project, including how to acquire raw data from NYC Open Data and build the local DuckDB database for analysis.

---

## 📊 Datasets

### 1. 2016 Yellow Taxi Trip Data
Loads from TLC parquet files directly.

### 2. Automated Traffic Volume Counts
1. Go to [NYC Open Data - Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/about_data)
2. Select **Export** in the upper right corner
3. Download the CSV file

### 3. Motor Vehicle Collisions - Crashes
1. Go to [NYC Open Data - Motor Vehicle Collisions](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data)
2. Select **Export** in the upper right corner
3. Download the CSV file

---

## 🔧 Loading Data

### Conda Environment Setup

1. Create a new conda environment:
```bash
conda create -n nyc-taxi python=3.9
conda activate nyc-taxi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Setting up Local Persistent DuckDB

Run the data loading script:

```bash
python data/load_data.py
```

You will now have a DuckDB file in your local directory.

> **Note:** If needed, we may store the DuckDB file elsewhere (e.g., S3 bucket) if we are actively making changes that need to build off of one another dynamically and can't depend on each person running scripts independently.

---

## 🔀 Merging Data

> **⚠️ IN PROGRESS!**

Run the merge script:

```bash
python data/merge_data.py
```

---

## 📁 Project Structure

```
.
├── data/
│   ├── load_data.py          # Data loading script
│   ├── merge_data.py          # Data merging script (in progress)
│   └── raw/                   # Raw CSV files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError` when running `load_data.py`
- **Solution:** Ensure all CSV files are downloaded and placed in the correct directory

**Issue:** DuckDB connection errors
- **Solution:** Check that you have write permissions in the directory

**Issue:** Memory errors during data loading
- **Solution:** Consider processing data in chunks or increasing available memory

---

## 📝 Next Steps

After setting up the database:
1. Verify data integrity with basic queries
2. Explore the merged dataset structure
3. Proceed to the GNN model training phase

---

## 🤝 Contributing

If you encounter issues with data setup or have improvements to suggest, please open an issue or submit a pull request.

---

## 📧 Support

For questions about data setup, please [open an issue](../../issues) in the repository.