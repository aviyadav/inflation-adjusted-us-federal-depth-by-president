from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import requests
import streamlit as st

from nominal_federal_debt_increase_analysis import (
    ADMINISTRATIONS,
    LABEL_OFFSETS,
    PARTY_COLORS,
    adjust_debt_to_1969_dollars,
    assemble_chart_data,
    download_annual_cpi,
    download_treasury_debt,
)

st.set_page_config(
    page_title="U.S. federal debt by president",
    page_icon=":material/account_balance:",
    layout="wide",
)

st.title("U.S. federal debt by president")
st.markdown(
    "Explore nominal and inflation-adjusted changes in federal debt during each "
    "administration, measured from the prior fiscal year's debt."
)


@st.cache_data(ttl="1h", show_spinner="Loading Treasury and CPI data...")
def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    debt = download_treasury_debt()
    real_debt = adjust_debt_to_1969_dollars(debt, download_annual_cpi())
    return debt, assemble_chart_data(debt), assemble_chart_data(real_debt)


def make_debt_chart(
    chart_data: pl.DataFrame, *, inflation_adjusted: bool = False
) -> go.Figure:
    figure = go.Figure()
    parties_in_legend = set()

    for administration in ADMINISTRATIONS:
        label = administration["label"]
        party = administration["party"]
        series = chart_data.filter(pl.col("president") == label).sort("fiscal_year")
        if series.is_empty():
            continue

        color = PARTY_COLORS[party]
        figure.add_trace(
            go.Scatter(
                x=series["fiscal_year"].to_list(),
                y=series["increase_trillions"].to_list(),
                mode="lines+markers",
                name=party,
                legendgroup=party,
                showlegend=party not in parties_in_legend,
                line={"color": color, "width": 3},
                marker={"color": color, "size": 7},
                hovertemplate=(
                    f"{label}<br>FY %{{x}}<br>Debt increase: "
                    "$%{y:.3f}T<extra></extra>"
                ),
            )
        )
        parties_in_legend.add(party)

        last = series.row(-1, named=True)
        offset = LABEL_OFFSETS.get(label, (0, 8))
        figure.add_annotation(
            x=last["fiscal_year"],
            y=last["increase_trillions"],
            text=f"<b>{label}</b>",
            showarrow=False,
            xshift=offset[0],
            yshift=offset[1],
            xanchor="right" if offset[0] < 0 else "center",
            yanchor="bottom" if offset[1] >= 0 else "top",
            font={"color": "#222222", "size": 12},
        )

    earliest = int(chart_data.select(pl.col("fiscal_year").min()).item())
    latest = int(chart_data.select(pl.col("fiscal_year").max()).item())
    min_increase = float(chart_data.select(pl.col("increase_trillions").min()).item())
    max_increase = float(chart_data.select(pl.col("increase_trillions").max()).item())
    first_decade = ((earliest + 9) // 10) * 10
    decade_ticks = list(range(first_decade, latest + 1, 10))
    if latest not in decade_ticks:
        decade_ticks.append(latest)

    figure.update_layout(
        height=600,
        margin={"l": 20, "r": 30, "t": 60, "b": 20},
        legend={"orientation": "h", "y": 1.1, "x": 0.5, "xanchor": "center"},
        legend_title_text="Party",
        hovermode="closest",
        xaxis={
            "title": "Fiscal year",
            "range": [earliest - 1, latest + 2],
            "tickmode": "array",
            "tickvals": sorted(set(decade_ticks)),
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": (
                "Debt increase (1969 dollars)"
                if inflation_adjusted
                else "Debt increase"
            ),
            "range": [
                min(
                    -0.05 if inflation_adjusted else -0.25,
                    min_increase - (0.05 if inflation_adjusted else 0.25),
                ),
                max(
                    1.5 if inflation_adjusted else 10.0,
                    max_increase + (0.1 if inflation_adjusted else 0.8),
                ),
            ],
            "tickprefix": "$",
            "ticksuffix": "T",
            "tickformat": ",.1f" if inflation_adjusted else ",.0f",
            "dtick": 0.5 if inflation_adjusted else 1,
            "gridcolor": "#dddddd",
            "zerolinecolor": "#aaaaaa",
        },
    )
    return figure


try:
    debt, chart_data, real_chart_data = load_data()
except (requests.RequestException, RuntimeError, ValueError) as exc:
    st.error(f"Unable to load Treasury or CPI data: {exc}")
    st.stop()

latest_debt = debt.row(-1, named=True)
latest_change = chart_data.row(-1, named=True)
with st.container(horizontal=True):
    st.metric(
        "Latest nominal federal debt",
        f"${latest_debt['debt_dollars'] / 1e12:.3f}T",
        border=True,
    )
    st.metric("Latest fiscal year", str(latest_debt["fiscal_year"]), border=True)
    st.metric(
        f"{latest_change['president']} nominal debt increase",
        f"${latest_change['increase_trillions']:.3f}T",
        border=True,
    )

with st.container(border=True):
    st.subheader("Actual (nominal) change in U.S. federal debt by president")
    st.plotly_chart(
        make_debt_chart(chart_data),
        alt="Nominal federal debt increase during each presidential administration since Nixon",
        config={"displaylogo": False},
    )

st.caption(
    f"Latest Treasury record: {latest_debt['record_date']}. "
    "Fiscal years are attributed to the president taking office in the calendar "
    "year in which the fiscal year ends."
)

with st.container(border=True):
    st.subheader(
        "Inflation-adjusted change in U.S. federal debt by president, in 1969 dollars"
    )
    st.plotly_chart(
        make_debt_chart(real_chart_data, inflation_adjusted=True),
        alt="Inflation-adjusted federal debt change in 1969 dollars by presidential administration",
        config={"displaylogo": False},
    )

st.caption(
    "Real debt values use the annual average of the not-seasonally-adjusted CPI-U "
    "and are expressed in average 1969 dollars. Each line begins at the "
    "inflation-adjusted debt level for the prior fiscal year."
)

summary = (
    chart_data.sort("fiscal_year")
    .group_by(["president", "party"], maintain_order=True)
    .tail(1)
    .select("president", "party", "fiscal_year", "increase_trillions")
    .rename(
        {
            "president": "President",
            "party": "Party",
            "fiscal_year": "Final FY",
            "increase_trillions": "Debt increase ($T)",
        }
    )
)
with st.expander("Nominal administration summary"):
    st.dataframe(
        summary,
        hide_index=True,
        alt="Final debt increase for each presidential administration",
        column_config={
            "Debt increase ($T)": st.column_config.NumberColumn(format="$%.3fT")
        },
    )

st.caption(
    "Sources: U.S. Treasury, Fiscal Data, Historical Debt Outstanding; and U.S. "
    "Bureau of Labor Statistics CPI-U via FRED."
)
