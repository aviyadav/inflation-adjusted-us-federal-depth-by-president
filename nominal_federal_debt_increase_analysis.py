from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import requests
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Importing Libraries and Assigning Constants

TREASURY_API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
                "/v2/accounting/od/debt_outstanding")
OUTPUT_FILE = "nominal_us_federal_debt_increase_by_president.png"
INFLATION_ADJUSTED_OUTPUT_FILE = "inflation_adjusted_us_federal_debt_by_president.png"
FRED_CPI_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PARTY_COLORS = {"Republican": "#d9534f", "Democratic": "#3478c5"}

ADMINISTRATIONS = [
    {"label": "Nixon", "party": "Republican", "first_fy": 1969, "last_fy": 1974},
    {"label": "Ford", "party": "Republican", "first_fy": 1975, "last_fy": 1976},
    {"label": "Carter", "party": "Democratic", "first_fy": 1977, "last_fy": 1980},
    {"label": "Reagan", "party": "Republican", "first_fy": 1981, "last_fy": 1988},
    {"label": "Bush 41", "party": "Republican", "first_fy": 1989, "last_fy": 1992},
    {"label": "Clinton", "party": "Democratic", "first_fy": 1993, "last_fy": 2000},
    {"label": "Bush 43", "party": "Republican", "first_fy": 2001, "last_fy": 2008},
    {"label": "Obama", "party": "Democratic", "first_fy": 2009, "last_fy": 2016},
    {"label": "Trump", "party": "Republican", "first_fy": 2017, "last_fy": 2020},
    {"label": "Biden", "party": "Democratic", "first_fy": 2021, "last_fy": 2024},
    {"label": "Trump II", "party": "Republican", "first_fy": 2025, "last_fy": None},
]

LABEL_OFFSETS = {
    "Nixon": (0, 8), "Ford": (0, 8), "Carter": (0, 8), "Reagan": (-4, 8),
    "Bush 41": (0, 8), "Clinton": (0, 8), "Bush 43": (-4, 8), "Obama": (-10, -10),
    "Trump": (0, 8), "Biden": (0, 8), "Trump II": (0, 8),
}

# Defining a Function to Download the Treasury Data

def download_treasury_debt() -> pl.DataFrame:
    params = {"fields": "record_date,record_fiscal_year,debt_outstanding_amt",
              "filter": "record_fiscal_year:gte:1968", "sort":
              "record_fiscal_year", "page[size]": 500}
    res = requests.get(TREASURY_API, params=params, timeout=30)
    res.raise_for_status()
    data = res.json().get("data")
    if not data:
        raise RuntimeError("The Treasury API returned no debt data.")

    df = pl.DataFrame(pa.Table.from_pylist(data)).with_columns(
        pl.col("record_fiscal_year").cast(pl.Int64, strict=False).alias("fiscal_year"),
        pl.col("debt_outstanding_amt").cast(pl.Float64, strict=False).alias("debt_dollars"),
        pl.col("record_date").str.to_date(strict=False),
    )

    return (
        df.drop_nulls(subset=["fiscal_year", "debt_dollars"])
          .sort(["fiscal_year", "record_date"])
          .unique(subset=["fiscal_year"], keep="last", maintain_order=True)
          .sort("fiscal_year")
          .select("fiscal_year", "record_date", "debt_dollars")
    )

def download_annual_cpi() -> pl.DataFrame:
    res = requests.get(
        FRED_CPI_CSV,
        params={"id": "CPIAUCNS", "cosd": "1968-01-01"},
        timeout=30,
    )
    res.raise_for_status()
    if not res.content:
        raise RuntimeError("FRED returned no CPI data.")

    table = pacsv.read_csv(
        pa.BufferReader(res.content),
        convert_options=pacsv.ConvertOptions(null_values=["", "."]),
    )
    return (
        pl.DataFrame(table)
        .with_columns(
            pl.col("observation_date").cast(pl.Date).dt.year().alias("fiscal_year"),
            pl.col("CPIAUCNS").cast(pl.Float64, strict=False).alias("cpi"),
        )
        .drop_nulls(subset=["fiscal_year", "cpi"])
        .group_by("fiscal_year")
        .agg(pl.col("cpi").mean())
        .sort("fiscal_year")
    )


def adjust_debt_to_1969_dollars(
    debt: pl.DataFrame, annual_cpi: pl.DataFrame
) -> pl.DataFrame:
    base_cpi = annual_cpi.filter(pl.col("fiscal_year") == 1969)["cpi"]
    if base_cpi.is_empty():
        raise ValueError("No CPI value found for base year 1969.")

    adjusted = debt.join(annual_cpi, on="fiscal_year", how="left", validate="1:1")
    if missing := adjusted.filter(pl.col("cpi").is_null())["fiscal_year"].to_list():
        raise ValueError(f"Missing annual CPI values for fiscal years: {missing}")

    return (
        adjusted.with_columns(
            (pl.col("debt_dollars") * base_cpi[0] / pl.col("cpi")).alias("debt_dollars")
        )
        .select(debt.columns)
        .sort("fiscal_year")
    )


# Defining a Function to Assemble the Chart Data
def assemble_chart_data(debt: pl.DataFrame) -> pl.DataFrame:
    max_fy = int(debt.select(pl.col("fiscal_year").max()).item())
    assembled = []

    for adm in ADMINISTRATIONS:
        first_fy, last_fy = adm["first_fy"], adm["last_fy"]
        last_fy = max_fy if last_fy is None else min(last_fy, max_fy)
        baseline_fy = first_fy - 1

        if last_fy < first_fy:
            continue

        baseline_rows = debt.filter(
            pl.col("fiscal_year") == baseline_fy
        )["debt_dollars"]
        if baseline_rows.is_empty():
            raise ValueError(f"No Treasury debt value found for baseline "
                             f"FY {baseline_fy} ({adm['label']}).")

        series = debt.filter(pl.col("fiscal_year").is_between(baseline_fy, last_fy))
        expected = set(range(baseline_fy, last_fy + 1))
        if missing := sorted(expected - set(series["fiscal_year"])):
            raise ValueError(f"Missing fiscal years for {adm['label']}: {missing}")

        series = series.with_columns(
            ((pl.col("debt_dollars") - baseline_rows[0]) / 1e12)
            .alias("increase_trillions"),
            pl.lit(adm["label"]).alias("president"),
            pl.lit(adm["party"]).alias("party"),
        )
        assembled.append(series)

    if not assembled:
        raise RuntimeError("No administration series could be created.")
    return pl.concat(assembled)

# Defining a Function to Make the Chart

def make_chart(
    chart_data: pl.DataFrame,
    output_file: str = OUTPUT_FILE,
    *,
    inflation_adjusted: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    end_points = []

    for adm in ADMINISTRATIONS:
        label, party = adm["label"], adm["party"]
        series = chart_data.filter(pl.col("president") == label).sort("fiscal_year")
        if series.is_empty():
            continue

        color = PARTY_COLORS[party]
        ax.plot(series["fiscal_year"], series["increase_trillions"], color=color,
                linewidth=2.4, marker="o", markersize=3.8, zorder=3)

        first, last = series.row(0, named=True), series.row(-1, named=True)
        ax.scatter(first["fiscal_year"], first["increase_trillions"], s=42,
                   facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
        ax.scatter(last["fiscal_year"], last["increase_trillions"], s=46,
                   color=color, edgecolor="white", linewidth=0.8, zorder=5)
        end_points.append((label, float(last["fiscal_year"]),
                           float(last["increase_trillions"])))

    for label, x, y in end_points:
        offset = LABEL_OFFSETS.get(label, (0, 8))
        ax.annotate(label, xy=(x, y), xytext=offset, textcoords="offset points",
                    ha="right" if offset[0] < 0 else "center",
                    va="bottom" if offset[1] >= 0 else "top",
                    fontsize=10, fontweight="bold", color="#222222", zorder=6)

    earliest = int(chart_data.select(pl.col("fiscal_year").min()).item())
    latest = int(chart_data.select(pl.col("fiscal_year").max()).item())
    min_increase = float(
        chart_data.select(pl.col("increase_trillions").min()).item()
    )
    max_increase = float(
        chart_data.select(pl.col("increase_trillions").max()).item()
    )
    ax.set_xlim(earliest - 1, latest + 2)
    ax.set_ylim(
        min(
            -0.05 if inflation_adjusted else -0.25,
            min_increase - (0.05 if inflation_adjusted else 0.25),
        ),
        max(
            1.5 if inflation_adjusted else 10.0,
            max_increase + (0.1 if inflation_adjusted else 0.8),
        ),
    )
    title = (
        "Inflation-Adjusted Change in U.S. Federal Debt by President, in 1969 Dollars"
        if inflation_adjusted
        else "Actual (Nominal) Change in U.S. Federal Debt by President"
    )
    ax.set_title(title, fontsize=24, fontweight="bold", pad=48)
    if inflation_adjusted:
        ax.set_ylabel("Constant 1969 purchasing power ($ trillions)", fontsize=14)

    handles = [Line2D([0], [0], color=PARTY_COLORS[p], linewidth=3, label=p)
               for p in ["Republican", "Democratic"]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=2, frameon=False, fontsize=15, handlelength=2.2, columnspacing=2.2)

    # Dynamically calculate and set the x-axis tick marks to show decade
    # intervals, while ensuring the final data year is always included:
    first_decade = ((earliest + 9) // 10) * 10
    decade_ticks = list(range(first_decade, latest + 1, 10))
    if latest not in decade_ticks:
        decade_ticks.append(latest)
    ax.set_xticks(sorted(set(decade_ticks)))

    precision = 1 if inflation_adjusted else 0
    ax.yaxis.set_major_locator(MultipleLocator(0.5 if inflation_adjusted else 1))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: (
                "$0" if abs(v) < 1e-4
                else f"{'-' if v < 0 else ''}${abs(v):.{precision}f}T")))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=15,
                   colors="#666666")

    if inflation_adjusted:
        source_note = (
            "Sources: U.S. Treasury Fiscal Data and U.S. Bureau of Labor Statistics "
            "CPI-U via FRED.\n"
            "Debt values are expressed in average 1969 dollars using the "
            "not-seasonally-adjusted CPI-U.\n"
            "Each line begins at the inflation-adjusted debt level for the prior "
            "fiscal year."
        )
    else:
        source_note = (
            "Source: U.S. Treasury, Fiscal Data, Historical Debt Outstanding.\n"
            "Fiscal years are attributed to the president taking office in the "
            "calendar year in which the fiscal year ends. \n"
            "Each line begins at the prior fiscal year's debt."
        )
    fig.text(0.08, 0.026, source_note, ha="left", va="bottom", fontsize=12,
             color="#666666")
    plt.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.18)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)

# Bringing it all together

def main() -> None:
    debt = download_treasury_debt()
    chart_data = assemble_chart_data(debt)
    real_debt = adjust_debt_to_1969_dollars(debt, download_annual_cpi())
    real_chart_data = assemble_chart_data(real_debt)

    summary = (
        chart_data.sort("fiscal_year")
        .group_by(["president", "party"], maintain_order=True)
        .tail(1)
        .select("president", "party", "fiscal_year", "increase_trillions")
        .rename({"president": "President", "party": "Party",
                 "fiscal_year": "Final FY",
                 "increase_trillions": "Debt increase ($T)"})
        .with_columns(pl.col("Debt increase ($T)").round(3))
    )
    with pl.Config(tbl_rows=-1):
        print(summary)

    make_chart(chart_data)
    make_chart(
        real_chart_data,
        INFLATION_ADJUSTED_OUTPUT_FILE,
        inflation_adjusted=True,
    )
    print(f"\nCharts saved as: {OUTPUT_FILE} and {INFLATION_ADJUSTED_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
