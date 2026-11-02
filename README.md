# U.S. federal debt dashboard

An interactive Streamlit dashboard showing nominal and inflation-adjusted federal debt changes by presidential administration. It uses debt data from the U.S. Treasury Fiscal Data API and CPI-U data from FRED, with real values expressed in average 1969 dollars.

## Run

```sh
uv sync
uv run streamlit run app.py
```

Treasury and CPI data are cached for one hour. The standalone Matplotlib report generates both actual and inflation-adjusted charts with:

```sh
uv run python nominal_federal_debt_increase_analysis.py
```
