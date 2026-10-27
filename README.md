# Reading Mobile Money Data: Beginner Analytics with Pandas and Matplotlib

Prototype for the PyCon Tanzania 2026 talk. All data is **synthetic**: no real customer or provider data is used.

## Run it

```bash
pip install pandas numpy matplotlib
python mobile_money_demo.py
```

## What it does

| Phase | Talk section | What happens |
|---|---|---|
| 0 | Introduction | Generates a messy synthetic dataset (about 6,000 transactions over 120 days) with missing values, duplicates and inconsistent text case |
| 1 | Cleaning with Pandas | Inspects data, drops duplicates, parses timestamps, fixes text case, fills missing network, drops rows with no amount |
| 2 | Aggregation | groupby metrics: transaction counts by type, average cash-out fee, peak hour, value by type, weekday-by-hour counts |
| 3 | Visualization | Four charts saved in `charts/` |

## Files

- `mobile_money_demo.py`: the full pipeline. Sections marked `# %%` can be pasted into Jupyter cells.
- `raw_transactions.csv`: the messy input.
- `clean_transactions.csv`: the cleaned output.
- `charts/`: daily trend, value by type, peak hours, weekday-by-hour heatmap.

## Notes

- The data is generated with a fixed random seed, so results are reproducible.
- Fee rates (2% cash-out, 0.5% P2P) are illustrative assumptions, not real provider tariffs.
