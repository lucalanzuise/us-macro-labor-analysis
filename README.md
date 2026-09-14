# US Macroeconomic and Labour-Market Analysis

This repository contains a reproducible Python analysis of inflation and labour-market dynamics in the United States using data from the Federal Reserve Economic Data (FRED) database.

The project originated from graduate macroeconomics coursework and was subsequently consolidated, corrected and extended into a single reproducible analytical workflow.

## Main components

- Collection and validation of macroeconomic and labour-market time series
- Transformation of monthly and quarterly data
- Construction of job-finding and separation rates
- Beveridge-curve analysis across economic periods
- Phillips-curve estimation over different subsamples
- Rolling-window regression analysis
- One-quarter-ahead inflation forecasting
- Comparison with a naive forecast benchmark
- Automatic production of tables, figures and data-quality reports

## Data

The analysis uses publicly available FRED series covering inflation, unemployment, employment, vacancies, labour-force participation and other macroeconomic indicators. Series definitions, units, frequencies and transformations are documented within the program and generated metadata tables.

Raw data are downloaded automatically when the script is executed and are not stored permanently in this repository.

## Software

The analysis was developed in Python using:

- pandas
- NumPy
- Matplotlib

## Running the analysis

Install the required packages:

```bash
python -m pip install -r requirements.txt
