from django.core.management.base import BaseCommand, CommandParser

class Command(BaseCommand):
    help = "Fetches historical weather data from the NASA POWER API for a given location and date range."

    def add_arguments(self, parser: CommandParser) -> None:
        """
        Adds command-line arguments for latitude, longitude, start date, and end date.
        """
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

    def handle(self, *args, **options):
        """
        The main logic of the management command.
        This method is executed when the command is run.
        """
        lat = options["lat"]
        lon = options["lon"]
        start_date = options["start"]
        end_date = options["end"]

        self.stdout.write(
            self.style.SUCCESS(
                f"Starting data fetch for lat={lat}, lon={lon} from {start_date} to {end_date}..."
            )
        )

        # ------------------------------------------------------------------
        # TODO: Insert your data fetching logic from backend/main.py here.
        #
        # You can adapt the `fetch_power`, `json_to_dataframe`, and
        # `get_power_data` functions.
        #
        # Example of what you might do:
        #
        # try:
        #     df = get_power_data(lat, lon, start_date, end_date)
        #     self.stdout.write(self.style.SUCCESS(f"Successfully fetched {len(df)} rows of data."))
        #     # Further processing, like saving to a file or database...
        # except Exception as e:
        #     self.stderr.write(self.style.ERROR(f"An error occurred: {e}"))
        #
        # ------------------------------------------------------------------

        self.stdout.write(
            self.style.SUCCESS("Boilerplate command executed successfully.")
        )
        self.stdout.write(
            self.style.WARNING("Replace the placeholder logic with your actual data fetching code.")
        )
