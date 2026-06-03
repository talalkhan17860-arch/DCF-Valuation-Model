# DCF Valuation Model

A 3-stage Discounted Cash Flow valuation model in Python. It pulls live financials via Yahoo Finance, builds up WACC from CAPM, projects free cash flow to the firm through an explicit and a fade period to a Gordon-Growth terminal value, and outputs intrinsic value per share — cross-checked against EV/EBITDA and P/E comparables on a football-field chart, with a WACC × terminal-growth sensitivity table.

## Methodology (FCFF)

- **Stage 1 — explicit** — free cash flow projected at an analyst-supplied growth rate (years 1–N).
- **Stage 2 — fade** — growth decays linearly from the Stage 1 rate toward the terminal rate over a short fade period.
- **Stage 3 — terminal value** — Gordon Growth Model perpetuity on the final fade-year cash flow.

Enterprise value = sum of discounted cash flows + discounted terminal value. Equity value = enterprise value − net debt. Intrinsic value per share = equity value ÷ shares outstanding.

## WACC build-up

- Cost of equity via CAPM: risk-free rate + β × equity risk premium.
- After-tax cost of debt, estimated from interest expense over total debt.
- Blended at market weights (equity vs debt).

## Output

- Intrinsic value per share vs current market price, with implied upside/downside and margin of safety.
- EV/EBITDA and P/E cross-checks against sector-average multiples.
- Football-field chart summarising the valuation range across methods.
- WACC × terminal-growth sensitivity table.

## How to run

```bash
pip install yfinance numpy pandas matplotlib scipy
python dcf_valuation.py --ticker AAPL
python dcf_valuation.py --ticker MSFT --growth 0.10 --years 7
python dcf_valuation.py --ticker TSLA --growth 0.15 --years 7 --tgr 0.03
```

Flags: `--ticker`, `--growth` (Stage 1 FCF growth), `--years` (Stage 1 horizon), `--tgr` (terminal growth), `--fade` (fade-period length). The script prints a full report and saves a 6-panel dashboard as `<TICKER>_dcf_valuation.png`.

## Limitations

A DCF is only as good as its inputs, and this one is deliberately transparent rather than exhaustive:

- **High sensitivity to WACC and terminal growth** — a 1–2 point change in either moves intrinsic value by 20–30%, which is what the sensitivity table is there to show. The range matters more than the point estimate.
- **Relies on Yahoo Finance data**, which can be incomplete or inconsistent across tickers.
- **A single forward growth assumption** drives the explicit period — no bottom-up revenue build or three-statement linkage.
- **Standalone DCF misses buybacks, optionality and intangible value**, so for some companies (e.g. Apple) intrinsic value can sit well below market price — which is why the model triangulates against EV/EBITDA and P/E comparables.

## Tech stack

Python, NumPy, Pandas, Matplotlib, SciPy, yfinance.

## Author

Talal Bin Waqas
