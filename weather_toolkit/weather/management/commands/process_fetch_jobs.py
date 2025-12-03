from django.core.management.base import BaseCommand
from django.db import transaction
from time import sleep
from datetime import datetime

from weather.models import FetchJob
from weather.data_fetcher import fetch_and_store_data


class Command(BaseCommand):
    help = "Process pending FetchJob entries and perform data fetches."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process a single job and exit.",
        )
        parser.add_argument(
            "--poll-interval",
            type=int,
            default=5,
            help="Seconds to wait between polling for new jobs.",
        )

    def handle(self, *args, **options):
        once = options["once"]
        poll_interval = options["poll_interval"]

        self.stdout.write(self.style.SUCCESS("Starting FetchJob processor"))

        try:
            while True:
                job = None
                # Obtain a pending job in a transaction to avoid races
                with transaction.atomic():
                    job = (
                        FetchJob.objects.select_for_update(skip_locked=True)
                        .filter(status=FetchJob.STATUS_PENDING)
                        .order_by("created_at")
                        .first()
                    )
                    if job:
                        job.status = FetchJob.STATUS_RUNNING
                        job.attempts += 1
                        job.save()

                if not job:
                    if once:
                        self.stdout.write("No pending jobs found. Exiting.")
                        return
                    sleep(poll_interval)
                    continue

                self.stdout.write(f"Processing job {job.id} for {job.latitude},{job.longitude}")
                try:
                    ok = fetch_and_store_data(job.latitude, job.longitude, job.start_date, job.end_date)
                    job.status = FetchJob.STATUS_DONE if ok else FetchJob.STATUS_FAILED
                    job.last_error = None if ok else "fetch returned False"
                except Exception as e:
                    job.status = FetchJob.STATUS_FAILED
                    job.last_error = str(e)
                job.updated_at = datetime.now()
                job.save()

                if once:
                    self.stdout.write("Processed one job. Exiting.")
                    return

        except KeyboardInterrupt:
            self.stdout.write("Interrupted — exiting.")