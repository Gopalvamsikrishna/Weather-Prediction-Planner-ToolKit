import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from django.core.management.base import BaseCommand, CommandParser

# Assuming your model is defined in weather/models.py
from weather.models import HistoricalDataPoint

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

class Command(BaseCommand):
    help = "Fetches historical weather data from NASA POWER and saves it to the database."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--lat",
            type=float,
            required=True,
            help="Latitude of the location (e.g., 12.97).",
        )
        parser.add_argument(
            "--lon",
            type=float,
            required=True,
            help="Longitude of the location (e.g., 77.59).",
        )
        parser.add_argument(
            "--start",
            type=str,
            required=True,
            help="Start date in YYYYMMDD format (e.g., 19900101).",
        )
        parser.add_argument(
            "--end",
            type=str,
            required=True,
            help="End date in YYYYMMDD format (e.g., 20231231).",
        )
        parser.add_argument(
            "--chunk-days",
            type=int,
            default=365 * 5, # Fetch in 5-year chunks
            help="Chunk size in days for API calls.",
        )

    def handle(self, *args, **options):
        lat = options["lat"]
        lon = options["lon"]
        start_date = options["start"]
        end_date = options["end"]
        chunk_days = options["chunk_days"]

        self.stdout.write(
            self.style.SUCCESS(
                f"Starting data fetch for lat={lat}, lon={lon} from {start_date} to {end_date}..."
            )
        )

        params = ["T2M_MAX", "T2M_MIN", "T2M", "PRECTOTCORR", "WS10M", "RH2M"]

        dfs = []
        try:
            for s_chunk, e_chunk in chunk_date_ranges(
                start_date, end_date, days=chunk_days
            ):
                self.stdout.write(f"Fetching {s_chunk} -> {e_chunk} ...")
                j = fetch_power(lat, lon, s_chunk, e_chunk, params)
                df_chunk = json_to_dataframe(j, params)
                dfs.append(df_chunk)

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to fetch data: {e}"))
            return

        if not dfs:
            self.stdout.write(self.style.WARNING("No data fetched for the requested range."))
            return

        # Combine chunks, sort, and remove duplicates
        df = pd.concat(dfs)
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)
        df.reset_index(inplace=True) # Convert date index to column

        self.stdout.write(self.style.SUCCESS(f"Successfully fetched {len(df)} total records."))
        self.stdout.write("Saving data to the database...")

        # --- Save to Django Model ---
        # This assumes your HistoricalDataPoint model has fields that match the
        # column names from the dataframe.
        records_created = 0
        records_updated = 0

        for _, row in df.iterrows():
            # Prepare a dictionary of the data, renaming columns to match model fields if necessary
            # Example: `t2m_max=row["T2M_MAX"]`
            data_dict = {
                "latitude": lat,
                "longitude": lon,
                "date": row["date"],
                "year": row["date"].year,
                "day_of_year": row["date"].timetuple().tm_yday,
                "t2m_max": row.get("T2M_MAX"),
                "t2m_min": row.get("T2M_MIN"),
                "t2m": row.get("T2M"),
                "prectotcorr": row.get("PRECTOTCORR"),
                "ws10m": row.get("WS10M"),
                "rh2m": row.get("RH2M"),
            }

            # Use update_or_create to avoid duplicates for the same location and date
            obj, created = HistoricalDataPoint.objects.update_or_create(
                latitude=lat, longitude=lon, date=row["date"], defaults=data_dict
            )

            if created:
                records_created += 1
            else:
                records_updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Database sync complete. Created: {records_created}, Updated: {records_updated}."
            )
        )
