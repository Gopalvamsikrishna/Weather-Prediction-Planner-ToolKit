import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from filelock import FileLock, Timeout

from .models import HistoricalDataPoint

class FetchInProgressError(Exception):
    """Custom exception to indicate that a fetch is already in progress."""
    pass

# --- NASA POWER API Configuration and Fetching Logic ---

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_power(
    lat, lon, start, end, params, community="AG", fmt="JSON", retries=3, backoff=2
):
    q = {
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "parameters": ",".join(params),
        "community": community,
        "format": fmt,
    }

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(POWER_BASE, params=q, timeout=60)
            if r.status_code != 200:
                print(f"ERROR: Request failed (status {r.status_code}) for {r.url}")
                print("Response body:", r.text)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            status = getattr(e.response, "status_code", None)
            if status and 400 <= status < 500:
                print(f"HTTP Error {status} (no retry): {e}")
                print(
                    "Response body:",
                    e.response.text if e.response is not None else "(no body)",
                )
                raise
            print(
                f"HTTP error on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds..."
            )
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(
                f"Request exception on attempt {attempt}/{retries}: {e}. Retrying in {backoff} seconds..."
            )
        time.sleep(backoff)
        backoff *= 2

    raise last_exc if last_exc is not None else RuntimeError(
        "Unknown error fetching POWER data"
    )


def json_to_dataframe(j, params):
    param_block = j.get("properties", {}).get("parameter", {})
    if not param_block:
        raise ValueError("No parameter block found in JSON response.")

    first = next(iter(param_block.values()))
    dates = sorted(first.keys())

    rows = []
    for d in dates:
        row = {"date": datetime.strptime(d, "%Y%m%d").date()}
        for p in params:
            val = param_block.get(p, {}).get(d, None)
            if val is None:
                row[p] = float("nan")
            else:
                try:
                    row[p] = float(val)
                except Exception:
                    row[p] = float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def chunk_date_ranges(start_str, end_str, days=365):
    start = datetime.strptime(start_str, "%Y%m%d").date()
    end = datetime.strptime(end_str, "%Y%m%d").date()
    cur_start = start
    while cur_start <= end:
        cur_end = min(end, cur_start + timedelta(days=days - 1))
        yield cur_start.strftime("%Y%m%d"), cur_end.strftime("%Y%m%d")
        cur_start = cur_end + timedelta(days=1)


def fetch_and_store_data(
    lat, lon, start_date, end_date, chunk_days=365 * 5, max_wait_seconds=60
):
    """
    Fetches data from the POWER API and stores it in the database.
    Includes a global locking mechanism to prevent any concurrent data fetches,
    which avoids database write conflicts with SQLite.

    Returns:
        bool: True if data was successfully fetched and stored, False otherwise.
    """
    lock_dir = Path("./run_locks")
    lock_dir.mkdir(exist_ok=True)

    # Per-location lock file to allow parallel fetches for different locations
    safe_lat = str(lat).replace(".", "_")
    safe_lon = str(lon).replace(".", "_")
    lock_file = lock_dir / f"fetch_{safe_lat}_{safe_lon}.lock"
    lock = FileLock(lock_file, timeout=1)

    try:
        # Non-blocking acquire: if another fetch for the same location is running,
        # skip and return False to indicate fetch is in progress.
        with lock.acquire(blocking=False):
            # Since we have the lock, we should double-check if data was
            # inserted by a process that held the lock just before us.
            if HistoricalDataPoint.objects.filter(latitude=lat, longitude=lon).exists():
                print(f"Data for {lat}, {lon} now exists. Skipping fetch.")
                return True

            print(
                f"Starting data fetch for lat={lat}, lon={lon} from {start_date} to {end_date}..."
            )

            params = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS10M", "RH2M"]

            dfs = []
            try:
                for s_chunk, e_chunk in chunk_date_ranges(
                    start_date, end_date, days=chunk_days
                ):
                    print(f"Fetching {s_chunk} -> {e_chunk} ...")
                    j = fetch_power(lat, lon, s_chunk, e_chunk, params)
                    df_chunk = json_to_dataframe(j, params)
                    dfs.append(df_chunk)
            except Exception as e:
                print(f"Failed to fetch data: {e}")
                return False

            if not dfs:
                print("No data fetched for the requested range.")
                return False

            df = pd.concat(dfs)
            df = df[~df.index.duplicated(keep="first")]
            df.sort_index(inplace=True)
            df.reset_index(inplace=True)

            print(f"Successfully fetched {len(df)} total records.")
            print("Saving data to the database...")

            # Use bulk_create for efficiency
            new_records = []
            for _, row in df.iterrows():
                new_records.append(
                    HistoricalDataPoint(
                        latitude=lat,
                        longitude=lon,
                        date=row["date"],
                        year=row["date"].year,
                        day_of_year=row["date"].timetuple().tm_yday,
                        t2m_max=row.get("T2M_MAX"),
                        t2m_min=row.get("T2M_MIN"),
                        t2m=row.get("T2M"),
                        prectotcorr=row.get("PRECTOTCORR"),
                        ws10m=row.get("WS10M"),
                        rh2m=row.get("RH2M"),
                    )
                )
            
            HistoricalDataPoint.objects.bulk_create(new_records, ignore_conflicts=True)
            
            print(f"Database sync complete for {lat}, {lon}.")
            return True

    except Timeout:
        print(f"A data fetch is already in progress. Skipping request for {lat}, {lon}.")
        return False
