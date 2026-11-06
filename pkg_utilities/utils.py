import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def numeric_diagnostics(df, columns):
    summaries = []

    fig, axes = plt.subplots(
        len(columns),
        2,
        figsize=(14, 3.5 * len(columns)),
        squeeze=False
    )

    for row, column in enumerate(columns):
        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            values.lt(lower_bound)
            | values.gt(upper_bound)
        )

        summaries.append({
            "feature": column,
            "count": values.size,
            "missing": df[column].isna().sum(),
            "mean": values.mean(),
            "median": values.median(),
            "skewness": values.skew(),
            "minimum": values.min(),
            "p01": values.quantile(0.01),
            "p99": values.quantile(0.99),
            "maximum": values.max(),
            "iqr_lower_bound": lower_bound,
            "iqr_upper_bound": upper_bound,
            "outlier_count": outliers.sum(),
            "outlier_pct": outliers.mean() * 100
        })

        sns.histplot(
            values,
            bins=30,
            kde=True,
            ax=axes[row, 0],
            color="#4C78A8"
        )

        axes[row, 0].axvline(
            lower_bound,
            color="red",
            linestyle="--",
            linewidth=1
        )

        axes[row, 0].axvline(
            upper_bound,
            color="red",
            linestyle="--",
            linewidth=1
        )

        axes[row, 0].set_title(
            f"Distribution of {column}"
        )

        sns.boxplot(
            x=values,
            ax=axes[row, 1],
            color="#73B6A2"
        )

        axes[row, 1].set_title(
            f"Box plot of {column}"
        )

    plt.tight_layout()

    summary = (
        pd.DataFrame(summaries)
        .set_index("feature")
    )

    return fig, summary