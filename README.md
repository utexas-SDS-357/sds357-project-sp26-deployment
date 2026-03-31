# Team Deployment — Examining Factors Associated with San Diego Police Searches

## Project Overview
This project studies various predictors of police stop outcomes, focusing on whether a search occurred during a stop.

We use stop-level data from the **Stanford Open Policing Project (SOPP)** in **San Diego**. The studied time period spans 2014-2017. We enrich the exisiting data set to include a geographic measure of social vulnerability. We use **Social Vulnerability Index (SVI)** data from the CDC.

## Data Description
### SOPP (Stanford Open Policing Project)
Stop-level records including:
- Demographics (e.g., subject age, race, sex)
- Situational characteristics (e.g., reason for stop, day/time)
- Outcome indicator (`search_conducted`)

### SVI (CDC/ATSDR Social Vulnerability Index)
This is our geographic vulnerability measure. We use the 2014 SVI tract-level data and use a spatial intersection to merge the data with the SOPP policing geography.

### Data folders
We keep data organized into:
- `data/raw/` — raw input files, including the original SOPP SVI, and policing zone data sets.
- `data/intermediate/` — intermediate data sets, which contain extraneous information that may be relevant for future directions.
- `data/final/` — data set used for modeling.

## Notebooks
We organize our pipeline into four Jupyter notebooks
- `0_data_wrangling.ipynb`: Merges SOPP and SVI data via spatial merging.
- `1_data_processing.ipynb`: Process merged data set and ensure data quality.
- `2_eda.ipynb`: Conduct exploratory data analysis.
- `3_modeling.ipynb`: Fit and evaluate machine learning models.

### Paths
The file `paths.py` defines shared path constants used by notebooks:
- `RAW_DATA_PATH`
- `INT_DATA_PATH`
- `FINAL_DATA_PATH`

This keeps file paths consistent across notebooks.

## Installation Instructions
We recommend using a virtual environment. Dependencies can be found in `requirements.txt`.
The exact versions used for analysis can be found in `requirements-lock.txt`.