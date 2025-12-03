from datetime import datetime

import pandas as pd

from weather.data_fetcher import json_to_dataframe, chunk_date_ranges


def test_json_to_dataframe_basic():
    sample = {
        "properties": {
            "parameter": {
                "T2M": {
                    "20230101": 20,
                    "20230102": 21,
                },
                "PRECTOTCORR": {
                    "20230101": 0.0,
                    "20230102": 1.2,
                },
            }
        }
    }

    df = json_to_dataframe(sample, ["T2M", "PRECTOTCORR"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "T2M" in df.columns
    assert "PRECTOTCORR" in df.columns


def test_chunk_date_ranges_multiple_chunks():
    # 3-day range with chunk size 2 should produce 2 chunks
    chunks = list(chunk_date_ranges("20230101", "20230103", days=2))
    assert len(chunks) == 2
    assert chunks[0] == ("20230101", "20230102")
    assert chunks[1] == ("20230103", "20230103")
