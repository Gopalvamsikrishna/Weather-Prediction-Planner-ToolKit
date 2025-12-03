from django.core.management.base import BaseCommand, CommandParser

from weather.data_fetcher import fetch_and_store_data


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
            default=365 * 5,  # Fetch in 5-year chunks
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

        try:
            fetch_and_store_data(lat, lon, start_date, end_date, chunk_days)
            self.stdout.write(self.style.SUCCESS("Data fetch and store completed successfully."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"An error occurred: {e}"))

