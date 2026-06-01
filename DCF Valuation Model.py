"""
============================================================
 Discounted Cash Flow (DCF) Valuation Model
 Author  : Talal Waqas
 Version : 3.0  —  Institutional Grade

 Methodology : Free Cash Flow to Firm (FCFF)
 Valuation   : 3-Stage DCF  +  EV/EBITDA  +  P/E  +  Football Field
 Output      : Bloomberg-style 6-panel dashboard  +  full CLI report

 Used in     : Investment Banking  |  Private Equity  |  Equity Research
============================================================

Usage:
    python dcf_valuation.py --ticker AAPL
    python dcf_valuation.py --ticker MSFT --growth 0.10 --years 7
    python dcf_valuation.py --ticker TSLA --growth 0.15 --years 7 --tgr 0.03
"""

import argparse
import warnings
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  Bloomberg-style dark colour palette
# ─────────────────────────────────────────────────────────────
DARK_BG      = "#0a0a0a"
PANEL_BG     = "#111111"
GRID_COL     = "#222222"
TEXT_PRIMARY = "#e8e8e8"
TEXT_DIM     = "#888888"
ACCENT_BLUE  = "#00aaff"
ACCENT_GREEN = "#00cc66"
ACCENT_RED   = "#ff3333"
ACCENT_GOLD  = "#ffcc00"
ACCENT_PURP  = "#aa66ff"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_PRIMARY,
    "axes.titlecolor":   TEXT_PRIMARY,
    "xtick.color":       TEXT_DIM,
    "ytick.color":       TEXT_DIM,
    "text.color":        TEXT_PRIMARY,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
})


# ======================================================================
#  HELPER
# ======================================================================

def _safe(df, key):
    """Extract a scalar from a yfinance DataFrame row, return None on failure."""
    try:
        val = df.loc[key].iloc[0]
        f   = float(val)
        return f if not np.isnan(f) else None
    except Exception:
        return None


def _info(d, key, default=0.0):
    """Extract a scalar from the yfinance info dict with a safe default."""
    try:
        v = d.get(key, default)
        return float(v) if v is not None else default
    except Exception:
        return default


# ======================================================================
#  DCF ENGINE
# ======================================================================

class DCFValuationModel:
    """
    Institutional-grade DCF valuation using FCFF methodology.

    Stages
    ------
    Stage 1  : High-growth phase  — analyst-supplied growth rate (Years 1–N)
    Stage 2  : Fade phase         — growth linearly decays to terminal rate (3 years)
    Stage 3  : Terminal value     — Gordon Growth Model (perpetuity)

    Cross-checks
    ------------
    • EV/EBITDA comparable multiples
    • P/E comparable multiples
    • 52-week trading range

    All four estimates are presented on a football field chart —
    the standard output in every IB pitch book.

    Parameters
    ----------
    ticker           : str   — Stock ticker (e.g. 'AAPL')
    growth_rate      : float — Stage 1 FCF growth rate  (e.g. 0.08 = 8%)
    terminal_growth  : float — Perpetual growth rate    (default 2.5%)
    projection_years : int   — Stage 1 horizon          (default 5)
    risk_free_rate   : float — 10Y US Treasury yield    (default 4.5%)
    equity_risk_prem : float — Market ERP               (default 5.5%)
    tax_rate         : float — Effective corporate tax  (default 21%)
    fade_years       : int   — Stage 2 fade period      (default 3)
    """

    def __init__(
        self,
        ticker:           str,
        growth_rate:      float = 0.08,
        terminal_growth:  float = 0.025,
        projection_years: int   = 5,
        risk_free_rate:   float = 0.045,
        equity_risk_prem: float = 0.055,
        tax_rate:         float = 0.21,
        fade_years:       int   = 3,
    ):
        self.ticker  = ticker.upper()
        self.g       = growth_rate
        self.tg      = terminal_growth
        self.years   = projection_years
        self.rf      = risk_free_rate
        self.erp     = equity_risk_prem
        self.tax     = tax_rate
        self.fade    = fade_years

        self.stock           = None
        self.info            = {}
        self.financials      = {}
        self.dcf_results     = {}
        self.sensitivity_table = None
        self.intrinsic_value = None
        self.wacc            = None

    # ──────────────────────────────────────────────────────────
    #  1. DATA FETCH
    # ──────────────────────────────────────────────────────────

    def fetch_data(self) -> dict:
        """Pull live financials from Yahoo Finance and compute WACC."""
        print(f"\n{'═'*64}")
        print(f"  Fetching live data for {self.ticker} ...")
        print(f"{'═'*64}")

        self.stock = yf.Ticker(self.ticker)
        self.info  = self.stock.info

        cf  = self.stock.cashflow
        inc = self.stock.income_stmt
        bal = self.stock.balance_sheet

        # ── Free Cash Flow ───────────────────────────────────
        try:
            op_cf = _safe(cf, "Operating Cash Flow")
            capex = _safe(cf, "Capital Expenditure")   # negative in yfinance
            fcf   = (op_cf + capex) if (op_cf and capex is not None) \
                    else _info(self.info, "freeCashflow")
        except Exception:
            fcf = _info(self.info, "freeCashflow")
        fcf = fcf or 0.0

        # ── EBITDA & EBIT ────────────────────────────────────
        ebitda = (_safe(inc, "EBITDA") or
                  _info(self.info, "ebitda") or
                  _info(self.info, "ebitdaMargins") * _info(self.info, "totalRevenue"))
        ebit   = _safe(inc, "EBIT") or (ebitda * 0.85 if ebitda else 0)
        nopat  = ebit * (1 - self.tax)

        # ── Revenue ──────────────────────────────────────────
        revenue = (_safe(inc, "Total Revenue") or
                   _info(self.info, "totalRevenue"))

        # ── EPS (for P/E cross-check) ────────────────────────
        eps_ttm = _info(self.info, "trailingEps", None)
        pe_fwd  = _info(self.info, "forwardPE", None)

        # ── Balance Sheet ────────────────────────────────────
        cash = (_safe(bal, "Cash And Cash Equivalents") or
                _safe(bal, "Cash Cash Equivalents And Short Term Investments") or
                _info(self.info, "totalCash"))
        cash = cash or 0.0

        total_debt = (_safe(bal, "Total Debt") or
                      _info(self.info, "totalDebt"))
        total_debt = total_debt or 0.0

        # ── Capital Structure ────────────────────────────────
        market_cap    = _info(self.info, "marketCap") or 0.0
        total_capital = market_cap + total_debt
        if total_capital == 0:
            ev            = _info(self.info, "enterpriseValue", 1e12)
            total_capital = ev
            market_cap    = ev * 0.80

        # ── Beta & Cost of Equity (CAPM) ─────────────────────
        beta           = max(0.1, min(_info(self.info, "beta", 1.0) or 1.0, 3.0))
        cost_of_equity = self.rf + beta * self.erp

        # ── Cost of Debt ─────────────────────────────────────
        try:
            int_exp      = _safe(inc, "Interest Expense")
            cost_of_debt = (abs(int_exp) / total_debt
                            if (int_exp and total_debt > 0) else 0.04)
        except Exception:
            cost_of_debt = 0.04
        cost_of_debt = max(0.02, min(cost_of_debt, 0.15))

        # ── WACC ─────────────────────────────────────────────
        we   = market_cap / total_capital
        wd   = total_debt / total_capital
        wacc = we * cost_of_equity + wd * cost_of_debt * (1 - self.tax)
        wacc = max(0.05, min(wacc, 0.20))
        self.wacc = wacc

        # ── Shares & Price ───────────────────────────────────
        price  = float(self.info.get("currentPrice") or
                       self.info.get("regularMarketPrice") or 0)
        shares = _info(self.info, "sharesOutstanding", None)
        if not shares:
            shares = market_cap / price if (market_cap > 0 and price > 0) else 1e9
        shares = max(shares, 1)

        # ── EV multiples (for cross-checks) ──────────────────
        ev_ebitda_mkt = _info(self.info, "enterpriseToEbitda", None)

        self.financials = {
            "Company":        self.info.get("longName", self.ticker),
            "Sector":         self.info.get("sector", "N/A"),
            "Industry":       self.info.get("industry", "N/A"),
            "Current Price":  price,
            "Market Cap":     market_cap,
            "FCF (TTM)":      fcf,
            "EBITDA":         ebitda or 0.0,
            "EBIT":           ebit or 0.0,
            "NOPAT":          nopat or 0.0,
            "Revenue":        revenue or 0.0,
            "EPS (TTM)":      eps_ttm or 0.0,
            "Fwd P/E":        pe_fwd or 0.0,
            "EV/EBITDA (mkt)":ev_ebitda_mkt or 0.0,
            "Beta":           beta,
            "Cost of Equity": cost_of_equity,
            "Cost of Debt":   cost_of_debt,
            "WACC":           wacc,
            "Total Debt":     total_debt,
            "Cash":           cash,
            "Net Debt":       total_debt - cash,
            "Shares Out":     shares,
            "Weight Equity":  we,
            "Weight Debt":    wd,
            "52WeekHigh":     _info(self.info, "fiftyTwoWeekHigh", price * 1.2),
            "52WeekLow":      _info(self.info, "fiftyTwoWeekLow",  price * 0.8),
        }

        return self.financials

    # ──────────────────────────────────────────────────────────
    #  2. 3-STAGE DCF ENGINE
    # ──────────────────────────────────────────────────────────

    def run_dcf(self) -> dict:
        """
        3-Stage DCF:
          Stage 1 — explicit FCF projection at growth_rate  (Years 1–N)
          Stage 2 — fade period, growth decays to tg        (N+1 to N+fade)
          Stage 3 — Gordon Growth terminal value            (Year N+fade+1 → ∞)
        """
        if not self.financials:
            self.fetch_data()

        fin    = self.financials
        fcf    = fin["FCF (TTM)"]
        wacc   = fin["WACC"]
        shares = fin["Shares Out"]

        # Guard: negative FCF
        if fcf <= 0:
            print(f"\n⚠  FCF is negative (${fcf:,.0f}). Using NOPAT as proxy.\n")
            fcf = max(fin["NOPAT"], abs(fcf) * 0.5)
            if fcf <= 0:
                raise ValueError(
                    "FCF and NOPAT both non-positive. Cannot run DCF. "
                    "Try a different ticker or supply manual inputs."
                )

        if wacc <= self.tg:
            raise ValueError(
                f"WACC ({wacc:.2%}) ≤ terminal growth ({self.tg:.2%}). "
                "Reduce terminal growth rate or increase WACC."
            )

        # ── Stage 1: Explicit projection ─────────────────────
        s1_fcfs = [fcf * (1 + self.g) ** yr for yr in range(1, self.years + 1)]

        # ── Stage 2: Fade ────────────────────────────────────
        fade_rates = np.linspace(self.g, self.tg, self.fade + 2)[1:-1]  # exclude endpoints
        s2_fcfs    = []
        last_fcf   = s1_fcfs[-1]
        for fr in fade_rates:
            last_fcf = last_fcf * (1 + fr)
            s2_fcfs.append(last_fcf)

        all_fcfs = s1_fcfs + s2_fcfs
        total_yrs = self.years + self.fade

        # ── Discount all FCFs ─────────────────────────────────
        pv_fcfs = [cf / (1 + wacc) ** (i + 1) for i, cf in enumerate(all_fcfs)]

        # ── Stage 3: Terminal Value ───────────────────────────
        terminal_fcf = all_fcfs[-1] * (1 + self.tg)
        terminal_val = terminal_fcf / (wacc - self.tg)
        pv_terminal  = terminal_val / (1 + wacc) ** total_yrs

        # ── Enterprise → Equity → Per Share ──────────────────
        total_pv        = sum(pv_fcfs) + pv_terminal
        net_debt        = fin["Net Debt"]
        equity_value    = total_pv - net_debt
        intrinsic_value = equity_value / shares
        self.intrinsic_value = intrinsic_value

        price            = fin["Current Price"]
        upside           = (intrinsic_value / price - 1) * 100 if price else 0
        margin_of_safety = (1 - price / intrinsic_value) * 100  if intrinsic_value else 0

        self.dcf_results = {
            "Stage1 FCFs":           s1_fcfs,
            "Stage2 FCFs":           s2_fcfs,
            "All FCFs":              all_fcfs,
            "PV FCFs":               pv_fcfs,
            "Sum PV FCFs":           sum(pv_fcfs),
            "Terminal Value":        terminal_val,
            "PV Terminal Value":     pv_terminal,
            "Enterprise Value":      total_pv,
            "Net Debt":              net_debt,
            "Equity Value":          equity_value,
            "Intrinsic Value/Share": intrinsic_value,
            "Current Price":         price,
            "Upside/Downside (%)":   upside,
            "Margin of Safety (%)":  margin_of_safety,
            "TV as pct of EV":       pv_terminal / total_pv * 100,
            "Total Years":           total_yrs,
        }

        return self.dcf_results

    # ──────────────────────────────────────────────────────────
    #  3. MULTIPLES CROSS-CHECKS
    # ──────────────────────────────────────────────────────────

    def multiples_valuation(self) -> dict:
        """
        Cross-check DCF with EV/EBITDA and P/E multiples.
        Uses sector-average multiples as benchmarks.

        Returns implied price ranges for the football field chart.
        """
        fin    = self.financials
        ebitda = fin["EBITDA"]
        eps    = fin["EPS (TTM)"]
        shares = fin["Shares Out"]
        nd     = fin["Net Debt"]

        sector = fin.get("Sector", "").lower()

        # Sector EV/EBITDA benchmarks (Damodaran Jan-2024 approximations)
        ev_ebitda_map = {
            "technology":          {"low": 15, "mid": 22, "high": 32},
            "financial services":  {"low": 10, "mid": 14, "high": 20},
            "healthcare":          {"low": 12, "mid": 18, "high": 26},
            "consumer cyclical":   {"low": 8,  "mid": 12, "high": 18},
            "consumer defensive":  {"low": 10, "mid": 14, "high": 20},
            "industrials":         {"low": 9,  "mid": 13, "high": 18},
            "energy":              {"low": 5,  "mid": 8,  "high": 12},
            "utilities":           {"low": 8,  "mid": 11, "high": 15},
            "real estate":         {"low": 14, "mid": 18, "high": 24},
            "communication services": {"low": 8, "mid": 13, "high": 20},
            "basic materials":     {"low": 7,  "mid": 10, "high": 15},
        }

        # Sector P/E benchmarks
        pe_map = {
            "technology":          {"low": 20, "mid": 28, "high": 40},
            "financial services":  {"low": 10, "mid": 14, "high": 20},
            "healthcare":          {"low": 15, "mid": 22, "high": 32},
            "consumer cyclical":   {"low": 12, "mid": 18, "high": 26},
            "consumer defensive":  {"low": 14, "mid": 19, "high": 26},
            "industrials":         {"low": 14, "mid": 19, "high": 26},
            "energy":              {"low": 8,  "mid": 12, "high": 18},
            "utilities":           {"low": 14, "mid": 18, "high": 24},
            "real estate":         {"low": 18, "mid": 25, "high": 35},
            "communication services": {"low": 12, "mid": 18, "high": 26},
            "basic materials":     {"low": 10, "mid": 14, "high": 20},
        }

        ev_mult = ev_ebitda_map.get(sector, {"low": 10, "mid": 15, "high": 22})
        pe_mult = pe_map.get(sector,        {"low": 12, "mid": 18, "high": 26})

        # EV/EBITDA implied price
        ev_range = {}
        if ebitda and ebitda > 0:
            for k, m in ev_mult.items():
                implied_ev    = ebitda * m
                implied_eq    = implied_ev - nd
                ev_range[k]   = implied_eq / shares
        else:
            ev_range = {"low": None, "mid": None, "high": None}

        # P/E implied price
        pe_range = {}
        if eps and eps > 0:
            for k, m in pe_mult.items():
                pe_range[k] = eps * m
        else:
            pe_range = {"low": None, "mid": None, "high": None}

        self.multiples = {
            "EV/EBITDA Range": ev_range,
            "P/E Range":       pe_range,
            "EV/EBITDA Mults": ev_mult,
            "P/E Mults":       pe_mult,
            "Sector Used":     sector or "generic",
        }

        return self.multiples

    # ──────────────────────────────────────────────────────────
    #  4. SENSITIVITY ANALYSIS
    # ──────────────────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        wacc_range: list = None,
        tgr_range:  list = None,
    ) -> pd.DataFrame:
        """
        WACC vs Terminal Growth Rate sensitivity table.
        Standard IB pitch book output.
        """
        if wacc_range is None:
            b = self.wacc
            wacc_range = [b - 0.02, b - 0.01, b, b + 0.01, b + 0.02]

        if tgr_range is None:
            tgr_range = [0.010, 0.015, 0.020, 0.025, 0.030]

        fin      = self.financials
        fcf      = max(fin["FCF (TTM)"], fin["NOPAT"], 1.0)
        shares   = fin["Shares Out"]
        net_debt = fin["Net Debt"]

        table = pd.DataFrame(
            index=[f"{w:.1%}" for w in wacc_range],
            columns=[f"{t:.1%}" for t in tgr_range],
        )

        for w in wacc_range:
            for t in tgr_range:
                if w <= t:
                    table.loc[f"{w:.1%}", f"{t:.1%}"] = np.nan
                    continue
                # 3-stage
                s1  = [fcf * (1 + self.g) ** yr for yr in range(1, self.years + 1)]
                frs = np.linspace(self.g, t, self.fade + 2)[1:-1]
                s2  = []
                lf  = s1[-1]
                for fr in frs:
                    lf = lf * (1 + fr)
                    s2.append(lf)
                all_cf  = s1 + s2
                tot_yrs = self.years + self.fade
                pv_f    = sum(c / (1 + w) ** (i + 1) for i, c in enumerate(all_cf))
                tv      = (all_cf[-1] * (1 + t)) / (w - t)
                pv_tv   = tv / (1 + w) ** tot_yrs
                price   = ((pv_f + pv_tv) - net_debt) / shares
                table.loc[f"{w:.1%}", f"{t:.1%}"] = round(price, 2)

        self.sensitivity_table = table.astype(float)
        return self.sensitivity_table

    # ──────────────────────────────────────────────────────────
    #  5. VISUALISATION  —  Bloomberg-style 6-panel dashboard
    # ──────────────────────────────────────────────────────────

    def visualize(self, save: bool = True) -> plt.Figure:
        """
        6-Panel Bloomberg-style dashboard:
          [0,0] FCF Bridge         — TTM base + Stage 1 projections + Stage 2 fade
          [0,1] Value Bridge       — PV FCFs vs PV Terminal Value
          [1,0] Sensitivity Heatmap— WACC × TGR intrinsic value grid
          [1,1] Football Field     — DCF vs EV/EBITDA vs P/E vs 52W range
          [2,0] WACC Build-Up      — waterfall decomposition of WACC
          [2,1] Scenario Analysis  — Bear / Base / Bull per-share values
        """
        res  = self.dcf_results
        fin  = self.financials
        mult = getattr(self, "multiples", None)

        iv    = res["Intrinsic Value/Share"]
        price = res["Current Price"]
        upside= res["Upside/Downside (%)"]

        colour_iv = ACCENT_GREEN if upside > 0 else ACCENT_RED

        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor(DARK_BG)

        title = (
            f"DCF VALUATION  ▪  {fin['Company']} ({self.ticker})  ▪  "
            f"Intrinsic Value: ${iv:.2f}  ▪  "
            f"Current Price: ${price:.2f}  ▪  "
            f"{'▲' if upside > 0 else '▼'} {upside:+.1f}%"
        )
        fig.suptitle(title, fontsize=13, fontweight="bold",
                     color=TEXT_PRIMARY, y=0.99)

        gs = gridspec.GridSpec(3, 2, figure=fig,
                               hspace=0.55, wspace=0.35,
                               top=0.95, bottom=0.05)

        # ─── helper to style axes ────────────────────────────
        def _style(ax, title_txt):
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title_txt, color=ACCENT_BLUE,
                         fontsize=10, fontweight="bold", pad=8)
            ax.tick_params(colors=TEXT_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_COL)
            ax.yaxis.grid(True, color=GRID_COL, linewidth=0.5)
            ax.set_axisbelow(True)

        total_yrs  = res["Total Years"]
        s1_len     = self.years
        s2_len     = self.fade
        all_fcfs   = res["All FCFs"]
        labels_all = ([f"Yr {i}" for i in range(1, s1_len + 1)] +
                      [f"F{i}" for i in range(1, s2_len + 1)])

        # ── Panel 0,0 : FCF Bridge ───────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        _style(ax1, f"3-STAGE FCF PROJECTIONS  (Stage 1 g={self.g:.1%}  →  TGR={self.tg:.1%})")

        cols_fcf = ([ACCENT_BLUE] * s1_len + [ACCENT_PURP] * s2_len)
        bars1    = ax1.bar(["Base"] + labels_all,
                           [fin["FCF (TTM)"] / 1e9] + [f / 1e9 for f in all_fcfs],
                           color=["#555555"] + cols_fcf,
                           edgecolor=DARK_BG, linewidth=0.4, width=0.7)
        for bar in bars1[1:]:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f"${h:.1f}B", ha="center", va="bottom",
                     fontsize=6.5, color=TEXT_PRIMARY)
        ax1.set_ylabel("FCF  ($ Billions)", color=TEXT_DIM, fontsize=8)

        p1 = mpatches.Patch(color=ACCENT_BLUE,  label=f"Stage 1  High-growth ({self.g:.1%})")
        p2 = mpatches.Patch(color=ACCENT_PURP,  label=f"Stage 2  Fade → {self.tg:.1%}")
        ax1.legend(handles=[p1, p2], fontsize=7,
                   facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_DIM)

        # ── Panel 0,1 : Value Bridge ─────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        _style(ax2, "ENTERPRISE VALUE BRIDGE")

        pv_s1  = sum(res["PV FCFs"][:s1_len])
        pv_s2  = sum(res["PV FCFs"][s1_len:])
        pv_tv  = res["PV Terminal Value"]
        labels2= ["PV Stage 1\nFCFs", "PV Stage 2\nFade FCFs", "PV Terminal\nValue"]
        vals2  = [pv_s1 / 1e9, pv_s2 / 1e9, pv_tv / 1e9]
        cols2  = [ACCENT_BLUE, ACCENT_PURP, ACCENT_GOLD]
        b2     = ax2.bar(labels2, vals2, color=cols2,
                         edgecolor=DARK_BG, linewidth=0.4, width=0.45)
        total_ev = sum(vals2)
        for bar, val in zip(b2, vals2):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total_ev * 0.01,
                     f"${val:.1f}B\n({val/total_ev*100:.0f}%)",
                     ha="center", va="bottom",
                     fontsize=8.5, fontweight="bold", color=TEXT_PRIMARY)
        ax2.set_ylabel("Value  ($ Billions)", color=TEXT_DIM, fontsize=8)
        ax2.text(0.97, 0.97,
                 f"TV = {res['TV as pct of EV']:.0f}% of EV",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=8, color=ACCENT_GOLD)

        # ── Panel 1,0 : Sensitivity Heatmap ──────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        _style(ax3, "SENSITIVITY  —  Intrinsic Value / Share  ($)")

        sens  = self.sensitivity_table
        data  = sens.values.astype(float)
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        cmap = LinearSegmentedColormap.from_list(
            "ib", [ACCENT_RED, "#555555", ACCENT_GREEN]
        )
        im = ax3.imshow(data, cmap=cmap, aspect="auto",
                        vmin=vmin, vmax=vmax)
        ax3.set_xticks(range(len(sens.columns)))
        ax3.set_yticks(range(len(sens.index)))
        ax3.set_xticklabels(sens.columns, fontsize=7.5, color=TEXT_DIM)
        ax3.set_yticklabels(sens.index,   fontsize=7.5, color=TEXT_DIM)
        ax3.set_xlabel("Terminal Growth Rate", fontsize=8, color=TEXT_DIM)
        ax3.set_ylabel("WACC",               fontsize=8, color=TEXT_DIM)

        for i in range(len(sens.index)):
            for j in range(len(sens.columns)):
                val = data[i, j]
                txt = f"${val:.0f}" if not np.isnan(val) else "—"
                col = "white" if abs(val - (vmin + vmax) / 2) < (vmax - vmin) * 0.3 \
                      else TEXT_PRIMARY
                ax3.text(j, i, txt, ha="center", va="center",
                         fontsize=8, fontweight="bold", color=col)

        cb = plt.colorbar(im, ax=ax3, shrink=0.85)
        cb.ax.yaxis.set_tick_params(color=TEXT_DIM, labelsize=7)

        # Highlight base case cell
        base_wacc_label = f"{self.wacc:.1%}"
        if base_wacc_label in list(sens.index):
            row_idx = list(sens.index).index(base_wacc_label)
            ax3.add_patch(plt.Rectangle(
                (-.5, row_idx - .5), len(sens.columns), 1,
                fill=False, edgecolor=ACCENT_GOLD, linewidth=2
            ))

        # ── Panel 1,1 : Football Field ────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        _style(ax4, "FOOTBALL FIELD  —  Valuation Range Summary")
        ax4.set_facecolor(PANEL_BG)

        methods = []
        ranges  = []
        colours = []

        # DCF (sensitivity min/max)
        dcf_lo = float(np.nanmin(data))
        dcf_hi = float(np.nanmax(data))
        methods.append("DCF\n(Sensitivity Range)")
        ranges.append((dcf_lo, dcf_hi))
        colours.append(ACCENT_BLUE)

        # EV/EBITDA
        if mult and mult["EV/EBITDA Range"]["low"] is not None:
            ev_lo = mult["EV/EBITDA Range"]["low"]
            ev_hi = mult["EV/EBITDA Range"]["high"]
            methods.append(f"EV/EBITDA\n({mult['EV/EBITDA Mults']['low']}x – {mult['EV/EBITDA Mults']['high']}x)")
            ranges.append((ev_lo, ev_hi))
            colours.append(ACCENT_GREEN)

        # P/E
        if mult and mult["P/E Range"]["low"] is not None:
            pe_lo = mult["P/E Range"]["low"]
            pe_hi = mult["P/E Range"]["high"]
            methods.append(f"P/E\n({mult['P/E Mults']['low']}x – {mult['P/E Mults']['high']}x)")
            ranges.append((pe_lo, pe_hi))
            colours.append(ACCENT_PURP)

        # 52-Week Range
        methods.append("52-Week\nTrading Range")
        ranges.append((fin["52WeekLow"], fin["52WeekHigh"]))
        colours.append(ACCENT_GOLD)

        for i, (method, (lo, hi), col) in enumerate(zip(methods, ranges, colours)):
            mid = (lo + hi) / 2
            ax4.barh(i, hi - lo, left=lo, height=0.5,
                     color=col, alpha=0.6, edgecolor=col, linewidth=1)
            ax4.text(lo - (hi - lo) * 0.02, i,
                     f"${lo:.0f}", ha="right", va="center",
                     fontsize=7.5, color=col, fontweight="bold")
            ax4.text(hi + (hi - lo) * 0.02, i,
                     f"${hi:.0f}", ha="left", va="center",
                     fontsize=7.5, color=col, fontweight="bold")

        ax4.axvline(price, color=ACCENT_RED,  linewidth=2,
                    linestyle="--", label=f"Current: ${price:.2f}")
        ax4.axvline(iv,    color=ACCENT_GREEN, linewidth=2,
                    linestyle="-",  label=f"DCF IV:  ${iv:.2f}")
        ax4.set_yticks(range(len(methods)))
        ax4.set_yticklabels(methods, fontsize=8, color=TEXT_DIM)
        ax4.set_xlabel("Share Price  ($)", fontsize=8, color=TEXT_DIM)
        ax4.legend(fontsize=7.5, facecolor=PANEL_BG,
                   edgecolor=GRID_COL, labelcolor=TEXT_DIM,
                   loc="lower right")

        # ── Panel 2,0 : WACC Build-Up ────────────────────────
        ax5 = fig.add_subplot(gs[2, 0])
        _style(ax5, f"WACC BUILD-UP  (Total WACC = {fin['WACC']:.2%})")

        rf_contrib  = fin["Weight Equity"] * self.rf
        erp_contrib = fin["Weight Equity"] * fin["Beta"] * self.erp
        debt_contrib= fin["Weight Debt"]   * fin["Cost of Debt"] * (1 - self.tax)

        wacc_labels = [
            f"Risk-Free Rate\n(Rf = {self.rf:.2%})",
            f"Equity Risk Premium\n(β×ERP = {fin['Beta']:.2f}×{self.erp:.2%})",
            f"After-Tax Cost of Debt\n(Wd×Kd×(1-t))",
        ]
        wacc_vals = [rf_contrib, erp_contrib, debt_contrib]
        wacc_cols = [ACCENT_BLUE, ACCENT_PURP, ACCENT_GOLD]

        bars5 = ax5.bar(wacc_labels, [v * 100 for v in wacc_vals],
                        color=wacc_cols, edgecolor=DARK_BG,
                        linewidth=0.4, width=0.5)
        for bar, val in zip(bars5, wacc_vals):
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{val:.2%}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
        ax5.set_ylabel("Contribution to WACC  (%)", color=TEXT_DIM, fontsize=8)
        ax5.axhline(fin["WACC"] * 100, color=ACCENT_RED,
                    linewidth=1.5, linestyle="--")
        ax5.text(2.3, fin["WACC"] * 100 + 0.05,
                 f"WACC = {fin['WACC']:.2%}",
                 fontsize=8, color=ACCENT_RED, fontweight="bold")

        # ── Panel 2,1 : Bull / Base / Bear Scenario ──────────
        ax6 = fig.add_subplot(gs[2, 1])
        _style(ax6, "SCENARIO ANALYSIS  —  Intrinsic Value / Share  ($)")

        sens_flat = data[~np.isnan(data)]
        p10  = float(np.percentile(sens_flat, 10))
        p50  = float(np.percentile(sens_flat, 50))
        p90  = float(np.percentile(sens_flat, 90))

        sc_labels = ["Bear Case\n(10th percentile)",
                     "Base Case\n(50th percentile)",
                     "Bull Case\n(90th percentile)"]
        sc_vals   = [p10, p50, p90]
        sc_cols   = [ACCENT_RED, ACCENT_BLUE, ACCENT_GREEN]

        b6 = ax6.bar(sc_labels, sc_vals, color=sc_cols,
                     edgecolor=DARK_BG, linewidth=0.4, width=0.45)
        for bar, val in zip(b6, sc_vals):
            ret = (val / price - 1) * 100 if price else 0
            ax6.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(sc_vals) * 0.01,
                     f"${val:.2f}\n({ret:+.0f}%)",
                     ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=TEXT_PRIMARY)

        ax6.axhline(price, color=ACCENT_GOLD, linewidth=1.5,
                    linestyle="--")
        ax6.text(2.3, price + max(sc_vals) * 0.01,
                 f"Mkt ${price:.2f}",
                 fontsize=7.5, color=ACCENT_GOLD, fontweight="bold")
        ax6.set_ylabel("Intrinsic Value / Share  ($)", color=TEXT_DIM, fontsize=8)

        if save:
            fname = f"{self.ticker}_dcf_valuation.png"
            try:
                plt.savefig(fname, dpi=300, bbox_inches="tight",
                            facecolor=DARK_BG)
                print(f"\n✅  Dashboard saved as '{fname}'")
            except Exception as e:
                print(f"\n⚠  Could not save chart: {e}")

        plt.close(fig)
        return fig


# ======================================================================
#  TERMINAL OUTPUT
# ======================================================================

def print_results(model: DCFValuationModel):
    fin  = model.financials
    res  = model.dcf_results
    mult = getattr(model, "multiples", None)

    bar = "═" * 64

    print(f"\n{bar}")
    print(f"  COMPANY OVERVIEW")
    print(bar)
    print(f"  {'Company':<38} {fin['Company']}")
    print(f"  {'Sector / Industry':<38} {fin['Sector']}  /  {fin['Industry']}")
    print(f"  {'Current Price':<38} ${fin['Current Price']:>10.2f}")
    print(f"  {'Market Capitalisation':<38} ${fin['Market Cap']/1e9:>9.1f}B")
    print(f"  {'Revenue (TTM)':<38} ${fin['Revenue']/1e9:>9.1f}B")
    print(f"  {'EBITDA (TTM)':<38} ${fin['EBITDA']/1e9:>9.1f}B")
    print(f"  {'FCF (TTM)':<38} ${fin['FCF (TTM)']/1e9:>9.1f}B")

    print(f"\n{bar}")
    print(f"  WACC BUILD-UP")
    print(bar)
    print(f"  {'Beta':<38} {fin['Beta']:>10.2f}x")
    print(f"  {'Risk-Free Rate':<38} {model.rf:>9.2%}")
    print(f"  {'Equity Risk Premium':<38} {model.erp:>9.2%}")
    print(f"  {'Cost of Equity  (CAPM: Rf + β×ERP)':<38} {fin['Cost of Equity']:>9.2%}")
    print(f"  {'Cost of Debt (pre-tax)':<38} {fin['Cost of Debt']:>9.2%}")
    print(f"  {'Cost of Debt (after-tax)':<38} {fin['Cost of Debt']*(1-model.tax):>9.2%}")
    print(f"  {'Weight: Equity':<38} {fin['Weight Equity']:>9.2%}")
    print(f"  {'Weight: Debt':<38} {fin['Weight Debt']:>9.2%}")
    print(f"  {'──── WACC ────':<38} {fin['WACC']:>9.2%}")

    print(f"\n{bar}")
    print(f"  3-STAGE DCF ASSUMPTIONS")
    print(bar)
    print(f"  {'Stage 1: Growth Rate':<38} {model.g:>9.2%}  (Years 1–{model.years})")
    print(f"  {'Stage 2: Fade Period':<38} {model.fade:>9} years  (to TGR)")
    print(f"  {'Stage 3: Terminal Growth Rate':<38} {model.tg:>9.2%}  (perpetuity)")
    print(f"  {'Discount Rate (WACC)':<38} {fin['WACC']:>9.2%}")

    print(f"\n{bar}")
    print(f"  DCF VALUATION RESULTS")
    print(bar)
    print(f"  {'PV of Stage 1 FCFs':<38} ${sum(res['PV FCFs'][:model.years])/1e9:>9.1f}B")
    print(f"  {'PV of Stage 2 Fade FCFs':<38} ${sum(res['PV FCFs'][model.years:])/1e9:>9.1f}B")
    print(f"  {'PV of Terminal Value':<38} ${res['PV Terminal Value']/1e9:>9.1f}B")
    print(f"  {'Terminal Value as % of EV':<38} {res['TV as pct of EV']:>9.1f}%")
    print(f"  {'Enterprise Value':<38} ${res['Enterprise Value']/1e9:>9.1f}B")
    print(f"  {'Less: Net Debt':<38} ${res['Net Debt']/1e9:>9.1f}B")
    print(f"  {'Equity Value':<38} ${res['Equity Value']/1e9:>9.1f}B")

    print(f"\n{bar}")
    print(f"  VALUATION SUMMARY")
    print(bar)
    iv    = res["Intrinsic Value/Share"]
    price = res["Current Price"]
    upside= res["Upside/Downside (%)"]
    mos   = res["Margin of Safety (%)"]
    signal= "🟢 UNDERVALUED" if upside > 0 else "🔴 OVERVALUED"
    print(f"  {'Intrinsic Value per Share (DCF)':<38} ${iv:>10.2f}")
    print(f"  {'Current Market Price':<38} ${price:>10.2f}")
    print(f"  {'Upside / Downside':<38} {upside:>+9.1f}%  {signal}")
    print(f"  {'Margin of Safety':<38} {mos:>9.1f}%")

    if mult:
        print(f"\n{bar}")
        print(f"  MULTIPLES CROSS-CHECKS  ({mult['Sector Used'].title()} benchmarks)")
        print(bar)
        ev_r = mult["EV/EBITDA Range"]
        pe_r = mult["P/E Range"]
        if ev_r["low"] is not None:
            print(f"  {'EV/EBITDA Low / Mid / High':<38} "
                  f"  ${ev_r['low']:>7.2f}  /  ${ev_r['mid']:>7.2f}  /  ${ev_r['high']:>7.2f}")
        else:
            print(f"  EV/EBITDA cross-check not available (EBITDA ≤ 0)")
        if pe_r["low"] is not None:
            print(f"  {'P/E Low / Mid / High':<38} "
                  f"  ${pe_r['low']:>7.2f}  /  ${pe_r['mid']:>7.2f}  /  ${pe_r['high']:>7.2f}")
        else:
            print(f"  P/E cross-check not available (EPS ≤ 0)")

    print(f"\n{bar}")
    print(f"  SENSITIVITY  —  Intrinsic Value / Share ($)")
    print(f"  Rows = WACC  |  Columns = Terminal Growth Rate")
    print(bar)
    print(model.sensitivity_table.to_string())
    print()
    print(f"  ★ Base case cell highlighted in gold on dashboard")
    print(f"\n{bar}\n")


# ======================================================================
#  ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DCF Valuation Model v3.0 — Talal Waqas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python dcf_valuation.py --ticker AAPL\n"
            "  python dcf_valuation.py --ticker MSFT --growth 0.10 --years 7\n"
            "  python dcf_valuation.py --ticker TSLA --growth 0.15 --tgr 0.03\n"
        ),
    )
    parser.add_argument("--ticker", type=str,   default="AAPL",
                        help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--growth", type=float, default=0.08,
                        help="Stage 1 FCF growth rate, e.g. 0.08 for 8pct (default: 0.08)")
    parser.add_argument("--years",  type=int,   default=5,
                        help="Stage 1 projection years (default: 5)")
    parser.add_argument("--tgr",    type=float, default=0.025,
                        help="Terminal growth rate (default: 0.025)")
    parser.add_argument("--fade",   type=int,   default=3,
                        help="Stage 2 fade years (default: 3)")
    args = parser.parse_args()

    model = DCFValuationModel(
        ticker           = args.ticker,
        growth_rate      = args.growth,
        terminal_growth  = args.tgr,
        projection_years = args.years,
        fade_years       = args.fade,
    )

    model.fetch_data()
    model.run_dcf()
    model.sensitivity_analysis()
    model.multiples_valuation()
    print_results(model)
    model.visualize(save=True)