from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

VIBRANT_COLORS = [
    "#2563EB", "#EF4444", "#22C55E", "#A855F7", "#F97316", "#06B6D4",
    "#EC4899", "#84CC16", "#8B5CF6", "#F59E0B", "#14B8A6", "#E11D48"
]


def _clean_xy(df: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.DataFrame:
    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    return out.dropna(subset=[x_col, y_col])


def _fit_label(x: np.ndarray, y: np.ndarray, y_title: str = "y") -> tuple[float, float, str]:
    slope, intercept = np.polyfit(x, y, 1)
    sign = "+" if intercept >= 0 else "-"
    label = f"{y_title} = {slope:.4f}x {sign} {abs(intercept):.4f}"
    return slope, intercept, label


def _pretty_series_name(name: str) -> str:
    text = str(name).replace("statistical ", "").replace("P50 / Median", "P50")
    return text[:48]


def add_group_trendlines(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    group_col: str = "series",
    y_title: str = "y",
    show_annotations: bool = True,
    max_equations: int = 6,
    max_x: float | None = None,
    color_map: dict[str, str] | None = None,
):
    clean = _clean_xy(df, x_col, y_col)
    if max_x is not None:
        clean = clean[clean[x_col] <= max_x]
    equation_lines: list[str] = []

    trace_color_by_name = {}
    for trace in fig.data:
        trace_name = str(getattr(trace, "name", ""))
        line = getattr(trace, "line", None)
        marker = getattr(trace, "marker", None)
        color = None
        if line is not None:
            color = getattr(line, "color", None)
        if color is None and marker is not None:
            color = getattr(marker, "color", None)
        if trace_name and color:
            trace_color_by_name[trace_name] = color

    for idx, (series_name, part) in enumerate(clean.groupby(group_col, sort=False)):
        if len(part) < 2 or part[x_col].nunique() < 2:
            continue
        x = part[x_col].astype(float).to_numpy()
        y = part[y_col].astype(float).to_numpy()
        slope, intercept, label = _fit_label(x, y, y_title)
        x_fit = np.array([np.nanmin(x), np.nanmax(x)])
        y_fit = slope * x_fit + intercept
        short_name = _pretty_series_name(series_name)
        color = None
        if color_map:
            color = color_map.get(str(series_name))
        if color is None:
            color = trace_color_by_name.get(str(series_name))
        if color is None:
            color = VIBRANT_COLORS[idx % len(VIBRANT_COLORS)]
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            name=f"{short_name} trend",
            line={"dash": "dash", "width": 3, "color": color},
            opacity=0.72,
            hovertemplate=f"{short_name}<br>{label}<extra></extra>",
        ))
        if show_annotations and len(equation_lines) < max_equations:
            equation_lines.append(f"<b>{short_name}</b>: {label}")

    if show_annotations and equation_lines:
        # Keep regression equations outside the plotting area so the fitted lines remain visible.
        # The chart layout helper detects this below-chart panel and reserves extra bottom margin.
        equation_text = "<b>Trendline Equations</b><br>" + "<br>".join(equation_lines)
        fig.add_annotation(
            x=0.0,
            y=-0.22,
            xref="paper",
            yref="paper",
            text=equation_text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            bgcolor="rgba(248,250,252,0.98)",
            bordercolor="rgba(15,23,42,0.22)",
            borderwidth=1,
            borderpad=9,
            font={"size": 12, "color": "#334155"},
        )
    return fig


def polish(fig: go.Figure, height: int = 650, legend: str = "right"):
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    has_equation_panel = any(getattr(a, "y", 0) is not None and getattr(a, "y", 0) < 0 for a in annotations)
    equation_line_count = 0
    if has_equation_panel:
        for ann in annotations:
            if getattr(ann, "y", 0) is not None and getattr(ann, "y", 0) < 0 and getattr(ann, "text", None):
                equation_line_count = max(equation_line_count, str(ann.text).count("<br>") + 1)

    bottom_margin = 70
    if has_equation_panel:
        bottom_margin = max(190, 34 * equation_line_count + 90)
        height = max(height, 700)

    layout_kwargs = dict(
        height=height,
        template="plotly_white",
        colorway=VIBRANT_COLORS,
        margin=dict(l=70, r=40, t=92, b=bottom_margin),
        title={"x": 0.01, "xanchor": "left", "font": {"size": 19}},
        legend_title_text="Series",
        hovermode="x unified",
        font=dict(size=13, color="#334155"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if legend == "bottom":
        legend_y = -0.34 if has_equation_panel else -0.18
        layout_kwargs["legend"] = dict(orientation="h", yanchor="top", y=legend_y, xanchor="left", x=0.0)
        layout_kwargs["margin"] = dict(l=70, r=40, t=92, b=max(bottom_margin, 180 if has_equation_panel else 120))
    else:
        layout_kwargs["legend"] = dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
        layout_kwargs["margin"] = dict(l=70, r=285, t=92, b=bottom_margin)
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(51,65,85,0.12)", zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(51,65,85,0.12)", zeroline=False, automargin=True)
    return fig


def pi_rank_chart(consolidated: pd.DataFrame, rows: int = 30):
    df = consolidated.sort_values("PI rank #").head(rows).copy()
    height = max(760, 24 * len(df) + 200)
    fig = px.bar(
        df, x="PI (@0.5% degr)", y="Name", orientation="h",
        hover_data=["State", "MWP", "Data yrs", "Degradation"],
        title=f"Top {len(df)} Sites by Lifetime Performance Index",
        color="PI (@0.5% degr)",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return polish(fig, height=height, legend="bottom")


def pi_distribution_by_type(plant: pd.DataFrame):
    df = plant.dropna(subset=["Type", "Lifetime PI"])
    fig = px.box(
        df, x="Type", y="Lifetime PI", color="Type", points="all", hover_data=["Name", "State", "MWp"],
        title="Lifetime PI Distribution by Structure",
        color_discrete_sequence=VIBRANT_COLORS,
    )
    fig.update_layout(showlegend=False)
    return polish(fig, height=600, legend="bottom")


def degradation_scatter(plant: pd.DataFrame):
    df = plant.dropna(subset=["Lifetime PI", "Lifetime Degr"])
    fig = px.scatter(
        df, x="Lifetime PI", y="Lifetime Degr", color="Type", size="MWp", hover_name="Name",
        hover_data=["State", "Type", "MWp", "1st full yr"],
        title="Lifetime PI vs. Degradation Result",
        color_discrete_sequence=VIBRANT_COLORS,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
    fig.update_yaxes(tickformat=".1%")
    return polish(fig, height=600, legend="right")


def annual_pi_cone(pi_long: pd.DataFrame, title: str = "Annual PI for the Longest-Running U.S. PV Systems"):
    # Use PV exceedance terminology: P10 is the stronger/high-performance case,
    # P50 is the median, and P90 is the weaker/downside case.
    grouped = pi_long.groupby("relative_year")["pi"].agg(
        p10=lambda ser: ser.quantile(0.90),
        p50="median",
        p90=lambda ser: ser.quantile(0.10),
        avg="mean",
        count="count"
    ).reset_index()
    plot_df = grouped.melt(
        id_vars=["relative_year", "count"],
        value_vars=["p10", "p50", "p90", "avg"],
        var_name="series",
        value_name="pi",
    )
    labels = {"p10": "P10 (high case)", "p50": "P50 / Median", "p90": "P90 (downside case)", "avg": "Average"}
    plot_df["series"] = plot_df["series"].map(labels)

    fig = px.line(
        plot_df,
        x="relative_year",
        y="pi",
        color="series",
        markers=True,
        title=title,
        color_discrete_map={
            "P10 (high case)": "#22C55E",
            "P50 / Median": "#2563EB",
            "P90 (downside case)": "#EF4444",
            "Average": "#F97316",
        },
        labels={"relative_year": "Relative operating year", "pi": "Performance Index"},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    trend_df = plot_df[plot_df["series"].isin(["P90 (downside case)", "P50 / Median"])]
    add_group_trendlines(
        fig, trend_df, "relative_year", "pi", "series", "PI", max_equations=2,
        max_x=16,
        color_map={"P50 / Median": "#2563EB", "P90 (downside case)": "#EF4444"},
    )
    fig.update_layout(legend_title_text="Fleet percentile")
    return polish(fig, height=760, legend="right")

def age_distribution_chart(plant: pd.DataFrame, actual_long: pd.DataFrame | None = None):
    df = plant.dropna(subset=["1st full yr"]).copy()
    if actual_long is not None and not actual_long.empty and "calendar_year" in actual_long.columns:
        latest_year = int(pd.to_numeric(actual_long["calendar_year"], errors="coerce").max())
    else:
        latest_year = int(pd.Timestamp.today().year)
    df["System age at latest available year"] = latest_year - pd.to_numeric(df["1st full yr"], errors="coerce") + 1
    df = df.dropna(subset=["System age at latest available year"])
    fig = px.histogram(
        df,
        x="System age at latest available year",
        nbins=max(8, min(25, df["System age at latest available year"].nunique())),
        color="Type" if "Type" in df.columns else None,
        title=f"U.S. Top 100 Oldest PV Systems: Age Distribution Through {latest_year}",
        labels={"System age at latest available year": "Operating age, years", "count": "Count"},
        color_discrete_sequence=VIBRANT_COLORS,
    )
    fig.update_layout(bargap=0.08)
    fig.update_yaxes(title_text="Count")
    return polish(fig, height=620, legend="right")


def state_count_chart(state_counts: pd.DataFrame):
    df = state_counts.sort_values("Count", ascending=True)
    height = max(840, 34 * len(df) + 190)
    fig = px.bar(
        df, x="Count", y="State", orientation="h",
        title="Number of Oldest-100 PV Sites by State",
        color="Count",
        color_continuous_scale="Sunsetdark",
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(tickmode="linear", automargin=True))
    return polish(fig, height=height, legend="bottom")



def piecewise_pi_trend_chart(df: pd.DataFrame):
    """Median and P90 PI plot with only segmented best-fit lines.

    The workbook table includes repeated segment rows. This function uses the
    observed Median and P90 curves as the source data, then computes the
    fitted lines separately for years 1-7 and 7-16. It intentionally avoids
    any single full-period fitted line.
    """
    observed_names = ["Median PI", "P90 statistical PI"]
    observed_color_map = {
        "Median PI": "#2563EB",
        "P90 statistical PI": "#EF4444",
    }
    trend_color_map = {
        ("Median PI", 1, 7): "#60A5FA",
        ("Median PI", 7, 16): "#1D4ED8",
        ("P90 statistical PI", 1, 7): "#F87171",
        ("P90 statistical PI", 7, 16): "#DC2626",
    }
    label_map = {
        "Median PI": "Median annual PI",
        "P90 statistical PI": "P90 downside annual PI",
    }
    observed = df[df["series"].isin(observed_names)].copy()
    observed["x"] = pd.to_numeric(observed["x"], errors="coerce")
    observed["y"] = pd.to_numeric(observed["y"], errors="coerce")
    observed = observed.dropna(subset=["x", "y"])

    fig = go.Figure()
    for series_name in observed_names:
        part = observed[observed["series"] == series_name].sort_values("x")
        if part.empty:
            continue
        # Keep the observed curves visually prominent, but omit them from the
        # legend so the legend only identifies the four segmented fits.
        fig.add_trace(go.Scatter(
            x=part["x"],
            y=part["y"],
            mode="lines+markers",
            name=label_map[series_name],
            showlegend=False,
            line={"width": 3, "color": observed_color_map[series_name]},
            marker={"size": 7},
            hovertemplate=(
                f"{label_map[series_name]}<br>"
                "Year %{x}<br>PI %{y:.3f}<extra></extra>"
            ),
        ))

    equation_lines: list[str] = []
    segments = [(1, 7), (7, 16)]
    for series_name in observed_names:
        part = observed[observed["series"] == series_name]
        for start, end in segments:
            seg = part[(part["x"] >= start) & (part["x"] <= end)].sort_values("x")
            if len(seg) < 2 or seg["x"].nunique() < 2:
                continue
            x = seg["x"].astype(float).to_numpy()
            y = seg["y"].astype(float).to_numpy()
            slope, intercept, label = _fit_label(x, y, "PI")
            x_fit = np.array([start, end], dtype=float)
            y_fit = slope * x_fit + intercept
            fit_label = f"{label_map[series_name]} fit, years {start}-{end}"
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=fit_label,
                line={
                    "width": 3,
                    "dash": "dash",
                    "color": trend_color_map[(series_name, start, end)],
                },
                opacity=0.86,
                hovertemplate=f"{fit_label}<br>{label}<extra></extra>",
            ))
            equation_lines.append(f"<b>{fit_label}</b>: {label}")

    if equation_lines:
        fig.add_annotation(
            x=0.0,
            y=-0.22,
            xref="paper",
            yref="paper",
            text="<b>Segmented best-fit equations</b><br>" + "<br>".join(equation_lines),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            bgcolor="rgba(248,250,252,0.98)",
            bordercolor="rgba(15,23,42,0.22)",
            borderwidth=1,
            borderpad=9,
            font={"size": 12, "color": "#334155"},
        )

    fig.update_layout(
        title="Median and P90 Annual PI Trends with Piecewise Operating Periods",
        xaxis_title="Year of operation",
        yaxis_title="Performance Index",
        legend_title_text="Segmented best-fit lines",
    )
    # Match the other workbook line charts: keep the equation panel below the
    # plotting area and the legend in the right-side legend column.
    return polish(fig, height=900, legend="right")

def workbook_line_chart(
    df: pd.DataFrame,
    title: str,
    y_title: str = "Performance Index",
    show_trendlines: bool = True,
    use_series_dash: bool = False,
):
    color_map = {
        "p10 (obs.rank)": "#22C55E",
        "P10 (obs.rank)": "#22C55E",
        "P10": "#22C55E",
        "Median": "#2563EB",
        "Median PI": "#2563EB",
        "P50 / Median": "#2563EB",
        "p90 (obs.rank)": "#EF4444",
        "P90 (obs.rank)": "#EF4444",
        "P90 statistical PI": "#EF4444",
        "P90": "#EF4444",
        "Median trend, years 1 to 7": "#2563EB",
        "Median trend, years 7 to 16": "#2563EB",
        "P90 trend, years 1 to 7": "#EF4444",
        "P90 trend, years 7 to 16": "#EF4444",
    }
    fig = px.line(
        df, x="x", y="y", color="series", markers=True, title=title,
        color_discrete_sequence=VIBRANT_COLORS, color_discrete_map=color_map,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    if use_series_dash:
        for trace in fig.data:
            if "trend" in str(trace.name).lower():
                trace.update(line=dict(width=3, dash="dash"), marker=dict(size=6))
    if show_trendlines:
        add_group_trendlines(fig, df, "x", "y", "series", y_title, color_map=color_map)
    fig.update_layout(xaxis_title="Year of operation", yaxis_title=y_title)
    return polish(fig, height=760, legend="right")


def workbook_ratio_chart(df: pd.DataFrame, title: str, show_trendlines: bool = True):
    df = df.copy()
    if "series" in df.columns:
        df["series"] = df["series"].replace({
            "p90/p50 ratio": "Empirical P90 downside / median",
            "p90/median ratio": "Statistical P90 downside / median (1.28σ)",
        })
    color_map = {
        "Years 1 to 7": "#2563EB",
        "Years 7 to 16": "#EF4444",
        "Empirical P90 downside / median": "#EF4444",
        "Statistical P90 downside / median (1.28σ)": "#2563EB",
    }
    fig = px.line(
        df, x="x", y="y", color="series", markers=True, title=title,
        color_discrete_sequence=VIBRANT_COLORS, color_discrete_map=color_map,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    if show_trendlines:
        add_group_trendlines(fig, df, "x", "y", "series", "Ratio", color_map=color_map)
    fig.update_layout(xaxis_title="Year of operation", yaxis_title="Ratio")
    return polish(fig, height=700, legend="right")


def workbook_bar_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str, x_title: str, y_title: str):
    fig = px.bar(
        df, x=x_col, y=y_col, title=title, text_auto=True, color=y_col,
        color_continuous_scale="Plasma",
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, coloraxis_showscale=False)
    return polish(fig, height=600, legend="bottom")


def temperature_comparison_chart(df: pd.DataFrame, title: str):
    plot_df = df.copy()
    fig = px.bar(
        plot_df,
        x="Temperature bin",
        y="Percent of sample",
        color="Set",
        barmode="group",
        text="Count",
        title=title,
        color_discrete_map={"Set A": "#2563EB", "Set B": "#F97316"},
        hover_data={"Count": True, "Percent of sample": ":.1%"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Module temperature bin",
        yaxis_title="Percent of sample",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(0.8, float(plot_df["Percent of sample"].max()) * 1.18)],
    )
    return polish(fig, height=620, legend="right")


def null_overlap_chart(df: pd.DataFrame, title: str = "Null Overlap Between Older-100 First 7 Years and Newer-25 Sample"):
    fig = px.line(
        df, x="x", y="y", color="series", markers=True, title=title,
        color_discrete_sequence=VIBRANT_COLORS,
        labels={"x": "Performance Index", "y": "Probability density"},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_layout(xaxis_title="Performance Index", yaxis_title="Probability density")
    return polish(fig, height=680, legend="right")


def variability_middle_systems_chart(detail: pd.DataFrame):
    rows = []
    for _, row in detail.iterrows():
        label = f"{int(row.get('PI rank'))} {row['Name']}" if pd.notna(row.get('PI rank')) else str(row['Name'])
        for year in range(1, 18):
            col = f"year_{year}"
            if col in detail.columns and pd.notna(row.get(col)):
                rows.append({
                    "Name": label,
                    "System": row["Name"],
                    "Relative year": year,
                    "PI": row[col],
                    "Normal POA variability": row.get("Normal POA Variability"),
                    "Lifetime variability": row.get("Lifetime variability"),
                    "Weather-explained share": row.get("Weather-explained share"),
                    "PI rank": row.get("PI rank"),
                })
    plot_df = pd.DataFrame(rows)
    fig = px.line(
        plot_df, x="Relative year", y="PI", color="Name", markers=False,
        title="Five Median-Ranked U.S. Systems (#48-#52 of 100): PI Histories",
        color_discrete_sequence=["#6F3CC3", "#0EA5E9", "#00B050", "#F97316", "#FF0000"],
        hover_data=["PI rank", "Normal POA variability", "Lifetime variability", "Weather-explained share"],
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        xaxis_title="Year of operation",
        yaxis_title="Performance Index",
        yaxis=dict(range=[0, 1.2]),
        title={"text": "Five Median-Ranked U.S. Systems (#48-#52 of 100): PI Histories<br><sup>All five systems have lifetime PI near 88%. Albuquerque and Fresno show large failures; the others follow steadier paths.</sup>"},
    )
    return polish(fig, height=640, legend="right")


def variability_single_system_chart(detail: pd.DataFrame, site_name: str):
    row = detail[detail["Name"] == site_name].iloc[0]
    rows = []
    for year in range(1, 18):
        pi_col = f"year_{year}"
        fit_col = f"fit_year_{year}"
        pi_val = row.get(pi_col)
        fit_val = row.get(fit_col)
        if pd.notna(pi_val) or pd.notna(fit_val):
            rows.append({
                "Relative year": year,
                "PI": float(pi_val) if pd.notna(pi_val) else np.nan,
                "Best-fit PI": float(fit_val) if pd.notna(fit_val) else np.nan,
            })
    plot_df = pd.DataFrame(rows).dropna(how="all", subset=["PI", "Best-fit PI"])
    poa_var = float(row.get("Normal POA Variability", np.nan))

    if plot_df["Best-fit PI"].isna().all():
        slope, intercept, _ = _fit_label(plot_df["Relative year"].to_numpy(), plot_df["PI"].to_numpy(), "PI")
        plot_df["Best-fit PI"] = slope * plot_df["Relative year"] + intercept
    else:
        fit_clean = plot_df.dropna(subset=["Best-fit PI"])
        slope, intercept, _ = _fit_label(fit_clean["Relative year"].to_numpy(), fit_clean["Best-fit PI"].to_numpy(), "PI")

    fig = go.Figure()

    if np.isfinite(poa_var):
        band = plot_df.dropna(subset=["Best-fit PI"]).copy()
        band["Upper weather range"] = band["Best-fit PI"] + poa_var
        band["Lower weather range"] = band["Best-fit PI"] - poa_var
        fig.add_trace(go.Scatter(
            x=band["Relative year"],
            y=band["Upper weather range"],
            mode="lines",
            name=f"Normal POA range, ±{poa_var:.1%}",
            line={"width": 0, "color": "rgba(34,197,94,0)"},
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=band["Relative year"],
            y=band["Lower weather range"],
            mode="lines",
            name=f"Normal POA range, ±{poa_var:.1%}",
            line={"width": 0, "color": "rgba(34,197,94,0)"},
            fill="tonexty",
            fillcolor="rgba(34,197,94,0.18)",
            hovertemplate="Normal POA weather range<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=plot_df["Relative year"], y=plot_df["PI"], mode="lines+markers",
        name=f"{site_name} observed PI", line=dict(width=3, color="#00B050"), marker=dict(size=7),
        hovertemplate="Year %{x}<br>Observed PI %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Relative year"], y=plot_df["Best-fit PI"], mode="lines",
        name="Best-fit PI trend",
        line=dict(width=3, dash="dash", color="#00B050"),
        hovertemplate="Best-fit PI %{y:.3f}<extra></extra>",
    ))

    weather_share = row.get("Weather-explained share")
    eq = f"PI = {slope:.4f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.4f}"
    note = (
        f"<b>Trend and weather-variability summary</b><br>"
        f"Best-fit trend: {eq}<br>"
        f"Observed lifetime variability: {row.get('Lifetime variability', np.nan):.1%}<br>"
        f"Normal POA variability band: ±{poa_var:.1%}<br>"
        f"Estimated share of observed scatter explained by normal POA variability: {weather_share:.0%}"
    )
    fig.add_annotation(
        x=0.0, y=-0.25, xref="paper", yref="paper", text=note, showarrow=False,
        xanchor="left", yanchor="top", align="left", bgcolor="rgba(248,250,252,0.98)",
        bordercolor="rgba(15,23,42,0.22)", borderwidth=1, borderpad=9, font={"size": 12, "color": "#334155"},
    )
    fig.update_layout(
        title=f"{site_name}: PI History with Normal POA Weather Range",
        xaxis_title="Year of operation",
        yaxis_title="Performance Index",
        yaxis=dict(range=[0, 1.2]),
    )
    return polish(fig, height=760, legend="bottom")

def site_actual_chart(actual_long: pd.DataFrame, site_name: str):
    df = actual_long[actual_long["Name"] == site_name].sort_values("relative_year")
    fig = px.line(
        df, x="calendar_year", y="actual_mwh", markers=True,
        title=f"Annual Generation, {site_name}",
        labels={"actual_mwh": "Actual MWh", "calendar_year": "Year"},
        color_discrete_sequence=VIBRANT_COLORS,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7), name="Annual generation")
    temp = df.assign(series="Annual generation", x=df["calendar_year"], y=df["actual_mwh"])
    add_group_trendlines(fig, temp, "x", "y", "series", "MWh", max_equations=1)
    return polish(fig, height=700, legend="right")


def site_pi_chart(pi_long: pd.DataFrame, site_name: str):
    df = pi_long[pi_long["Name"] == site_name].sort_values("relative_year")
    fig = px.line(
        df, x="relative_year", y="pi", markers=True,
        title=f"Relative-Year PI, {site_name}",
        labels={"pi": "PI", "relative_year": "Relative year"},
        color_discrete_sequence=VIBRANT_COLORS,
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7), name="Site PI")
    temp = df.assign(series="Site PI", x=df["relative_year"], y=df["pi"])
    add_group_trendlines(fig, temp, "x", "y", "series", "PI", max_equations=1)
    return polish(fig, height=700, legend="right")
