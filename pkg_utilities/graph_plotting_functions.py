import calendar
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns


def _set_month_ticks(axes, df, x_data, rotation):
    month_starts = df.loc[df[x_data].dt.day.eq(1), x_data]
    tick_step = max(1, math.ceil(len(month_starts) / 8))
    month_starts = month_starts.iloc[::tick_step]
    month_labels = [
        f"{calendar.month_name[date.month]} {date.year}"
        for date in month_starts
    ]
    axes.set_xticks(month_starts)
    axes.set_xticklabels(month_labels, rotation=rotation)


def plot_single(
    df, figsize, x_data, y_data, x_label, y_label,
    title, rotation, plot_type="bar", y_lim=None
):
    """Plot one DataFrame series against datetime-like x values.

    The input rows are plotted as provided, without sorting, filtering, or
    aggregation. Month ticks are placed only at existing dates on the first
    day of a month, with no more than eight labels to keep long date ranges
    readable. The returned Figure is left open for the caller to display,
    save, or close.
    """
    if plot_type not in {"bar", "line"}:
        raise ValueError("plot_type must be either 'bar' or 'line'")

    figure, axes = plt.subplots(figsize=figsize)

    if plot_type == "bar":
        axes.bar(df[x_data], df[y_data], width=1, label=y_data)
    else:
        axes.plot(df[x_data], df[y_data], label=y_data)

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.tick_params(axis="x", labelrotation=rotation)

    _set_month_ticks(axes, df, x_data, rotation)
    axes.legend()

    upper_limit = y_lim if y_lim else df[y_data].max() * 1.05
    axes.set_ylim(bottom=0, top=upper_limit)

    if df[y_data].dtype == int:
        axes.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    elif df[y_data].dtype == float:
        decimals = 0 if df[y_data].max() > 0.1 else 1
        axes.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=1, decimals=decimals)
        )

    return figure


def plot_secondary(
    df, figsize, x_data, y1_data, y2_data,
    x_label, y1_label, y2_label, title,
    rotation, y1_lim, y2_lim, y2_dec=2
):
    """Plot light-grey bars and up to 20 line series on separate y-axes.

    The bars sit behind the lines and span 80% of the median spacing between
    unique dates (at least one day). All input rows are plotted in their
    original order without aggregation, sorting, filtering, or modifying the
    DataFrame. Month-start ticks are selected only from dates present in the
    input and are thinned to at most eight labels for readability. The
    returned Figure remains open for the caller to display, save, or close.

    ``y2_data`` may be a column name or a non-empty list of column names.
    """
    secondary_columns = [y2_data] if isinstance(y2_data, str) else y2_data
    if not secondary_columns:
        raise ValueError("y2_data must be a column name or a non-empty list")
    if len(secondary_columns) > 20:
        print(
            "plot_secondary supports at most 20 secondary columns; "
            f"received {len(secondary_columns)}."
        )
        return None

    figure, left_axes = plt.subplots(figsize=figsize)
    right_axes = left_axes.twinx()

    left_axes.set_zorder(0)
    right_axes.set_zorder(1)
    right_axes.patch.set_visible(False)

    x_dates = pd.DatetimeIndex(df[x_data].dropna().unique()).sort_values()
    spacing_days = x_dates.to_series().diff().dt.total_seconds().div(86400)
    median_spacing = spacing_days.dropna().median()
    bar_width = max(1, median_spacing * 0.8) if pd.notna(median_spacing) else 1

    bars = left_axes.bar(
        df[x_data],
        df[y1_data],
        width=bar_width,
        label=y1_data,
        color="lightgrey",
        zorder=1,
    )
    colors = plt.get_cmap("tab20").colors
    lines = []
    for index, column in enumerate(secondary_columns):
        if index == 0:
            color = "C0"
        elif index == 1:
            color = "grey"
        else:
            color = colors[(index - 2) % len(colors)]
        (line,) = right_axes.plot(
            df[x_data],
            df[column],
            label=column,
            color=color,
            linewidth=2,
            zorder=2,
        )
        lines.append(line)

    left_axes.set_xlabel(x_label, fontsize=15)
    left_axes.set_ylabel(y1_label, fontsize=15)
    right_axes.set_ylabel(y2_label, fontsize=15)
    left_axes.set_title(title, fontsize=20, fontweight="bold")
    left_axes.tick_params(axis="x", labelrotation=rotation, labelsize=12)
    left_axes.tick_params(axis="y", labelsize=12)
    right_axes.tick_params(axis="x", labelrotation=rotation, labelsize=12)
    right_axes.tick_params(axis="y", labelsize=12)
    _set_month_ticks(left_axes, df, x_data, rotation)
    left_axes.tick_params(axis="x", labelsize=12)

    left_upper = y1_lim if y1_lim else df[y1_data].max() * 1.05
    right_upper = y2_lim if y2_lim else df[secondary_columns].max().max() * 1.05
    left_axes.set_ylim(bottom=0, top=left_upper)
    right_axes.set_ylim(bottom=0, top=right_upper)

    handles = [bars, *lines]
    labels = [handle.get_label() for handle in handles]
    right_axes.legend(handles, labels, loc="best", fontsize=12)

    if df[y1_data].dtype == int:
        left_axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
    elif df[y1_data].dtype == float:
        left_axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{value:.2%}")
        )

    if df[secondary_columns[0]].dtype == float:
        right_axes.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _: f"{value:.{y2_dec}%}")
        )

    for spine in right_axes.spines.values():
        spine.set_color("white")

    return figure




def plot_numeric_feature(
    df,
    column,
    bins=5,
    target="target_var"
):
    data = df[[column, target]].copy()

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    data = data.dropna(subset=[column, target])
    data = data[data[target].isin([0, 1])]

    data["target_label"] = data[target].map({
        0: "Non-defaulted",
        1: "Defaulted"
    })

    # Integer means quantile brackets; a list means custom boundaries
    if isinstance(bins, int):
        data["bracket"], bin_edges = pd.qcut(
            data[column],
            q=bins,
            duplicates="drop",
            retbins=True
        )
        threshold_name = "Quantile threshold"
    else:
        data["bracket"], bin_edges = pd.cut(
            data[column],
            bins=bins,
            include_lowest=True,
            retbins=True
        )
        threshold_name = "Bracket threshold"

    counts = (
        data.groupby(
            ["bracket", "target_label"],
            observed=True
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["Non-defaulted", "Defaulted"],
            fill_value=0
        )
    )

    default_rate = (
        counts["Defaulted"]
        / counts.sum(axis=1)
        * 100
    )

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2)

    ax_distribution = fig.add_subplot(grid[0, :])
    ax_counts = fig.add_subplot(grid[1, 0])
    ax_rate = fig.add_subplot(grid[1, 1])

    # KDE for continuous features; histogram for discrete features
    if data[column].nunique() > 15:
        sns.kdeplot(
            data=data,
            x=column,
            hue="target_label",
            common_norm=False,
            ax=ax_distribution
        )
    else:
        sns.histplot(
            data=data,
            x=column,
            hue="target_label",
            stat="density",
            common_norm=False,
            discrete=True,
            element="step",
            ax=ax_distribution
        )

    # Display the internal bin thresholds on the distribution
    for position, edge in enumerate(bin_edges[1:-1]):
        if abs(edge) < 1:
            edge_label = f"{edge:.4g}"
        else:
            edge_label = f"{edge:,.2f}".rstrip("0").rstrip(".")

        ax_distribution.axvline(
            edge,
            color="grey",
            linestyle="--",
            linewidth=1,
            alpha=0.8
        )

        ax_distribution.text(
            edge,
            0.97,
            edge_label,
            transform=ax_distribution.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color="dimgray"
        )

    ax_distribution.set_title(f"Distribution of {column}")

    # ax_distribution.set_title(f"Distribution of {column}")

    counts.plot.bar(
        ax=ax_counts,
        color=["#287D1E", "#3957D6"]
    )
    ax_counts.set_title("Observations by bracket")
    ax_counts.set_xlabel("")
    ax_counts.set_ylabel("Count")
    ax_counts.tick_params(axis="x", rotation=45)

    ax_rate.bar(
        default_rate.index.astype(str),
        default_rate,
        color="#73B6A2"
    )
    ax_rate.set_title("Default rate by bracket")
    ax_rate.set_ylabel("Default rate (%)")
    ax_rate.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    summary = counts.copy()
    summary["total"] = counts.sum(axis=1)
    summary["default_rate"] = default_rate

    return fig, summary