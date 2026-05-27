# Oldest 100 PV Systems Streamlit App

This reviewed version reproduces the workbook-derived logic using the calculated values stored in `data/100oldest.xlsx`. The public app does not allow users to upload, edit, or manipulate the source workbook.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main features

- Executive summary using workbook-derived Set A and Set B values
- Ranked Performance Index table
- Lifetime PI and degradation charts
- Relative-year PI spread chart with reviewer-limited trendlines through year 16
- State counts
- Clustered site map with click popups
- Site-level drilldown for metadata, annual generation, and PI trend

## Design choices

The app reads the workbook internally. It uses the workbook's existing calculated values first, rather than changing the methodology. Later versions can add EIA refresh, improved degradation models, Monte Carlo uncertainty, and additional regression methods after the workbook-matching version is approved.

## Review notes

See `REVIEW_FIXES_RESOLVED.md` for a point-by-point list of reviewer comments addressed in this version.
