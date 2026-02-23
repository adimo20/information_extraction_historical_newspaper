import pandas as pd
from typing import List, Tuple


def draw_stratified_sample(
    df: pd.DataFrame,
    intervals: List[Tuple[int, int]],
    zdb_ids: List[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Draws a stratified random sample:
    - 1 page per interval per newspaper
    - max. len(intervals) pages per newspaper
    """

    required_cols = {"publication_date", "zdb_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["publication_year"] = pd.to_datetime(
        df["publication_date"], errors="coerce"
    ).dt.year

    if zdb_ids is None:
        zdb_ids = sorted(df["zdb_id"].unique())

    samples = []

    for zdb_id in zdb_ids:
        df_paper = df[df["zdb_id"] == zdb_id]

        for i, (start, end) in enumerate(intervals):
            df_interval = df_paper[
                (df_paper["publication_year"] >= start) &
                (df_paper["publication_year"] <= end)
            ]

            if not df_interval.empty:
                samples.append(
                    df_interval.sample(
                        n=1,
                        random_state=random_state + i
                    )
                )

    return pd.concat(samples, ignore_index=True)
