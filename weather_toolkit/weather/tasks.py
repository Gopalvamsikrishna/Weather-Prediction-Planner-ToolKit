from celery import shared_task

from .data_fetcher import fetch_and_store_data


@shared_task(bind=True)
def fetch_data_task(self, lat: float, lon: float, start_date: str, end_date: str, chunk_days: int = 365 * 5):
    """Wrapper task that calls the synchronous fetch function.

    This keeps the existing fetch logic but runs it in a worker process.
    Returns True on success, False if fetch skipped or failed.
    """
    try:
        return fetch_and_store_data(lat, lon, start_date, end_date, chunk_days=chunk_days)
    except Exception as e:
        # Let exceptions bubble to Celery (they'll be logged) but return False
        raise
