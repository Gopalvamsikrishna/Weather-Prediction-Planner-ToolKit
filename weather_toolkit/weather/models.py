from django.db import models

class HistoricalDataPoint(models.Model):
    """
    Stores historical daily weather data retrieved from NASA POWER API.
    The data is indexed by latitude, longitude, and date.
    """
    # 1. Geospatial and Date Keys (Primary Index)
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)
    date = models.DateField(db_index=True) # The unique_together constraint handles uniqueness

    # Derived temporal fields (useful for quick filtering/grouping)
    year = models.IntegerField(db_index=True)
    day_of_year = models.IntegerField(db_index=True)

    # 2. Meteorological Variables (from NASA POWER params)
    # T2M_MAX, T2M_MIN, T2M are Temperature at 2m (Max, Min, Mean) (°C)
    t2m_max = models.FloatField(null=True)
    t2m_min = models.FloatField(null=True)
    t2m = models.FloatField(null=True)

    # PRECTOTCORR is Precipitation (mm/day)
    prectotcorr = models.FloatField(null=True)
    
    # WS10M is Wind Speed at 10m (m/s)
    ws10m = models.FloatField(null=True)
    
    # RH2M is Relative Humidity at 2m (%)
    rh2m = models.FloatField(null=True)

    # Note: 'heat_index' is calculated dynamically and doesn't need to be stored here.

    class Meta:
        # Define a composite index to ensure speed and uniqueness for the main query fields
        unique_together = (('latitude', 'longitude', 'date'),)
        ordering = ['date']
        verbose_name = "Historical Data Point"
        verbose_name_plural = "Historical Data Points"

    def __str__(self):
        return f"{self.date} at {self.latitude:.2f}, {self.longitude:.2f}"


class DataFetchLock(models.Model):
    """
    A lock to prevent concurrent data fetches for the same location.
    """
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)
    is_locked = models.BooleanField(default=True)
    locked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (('latitude', 'longitude'),)


class FetchJob(models.Model):
    """
    A simple DB-backed job queue entry for scheduling data fetches
    without external workers. Use `manage.py process_fetch_jobs` to
    run pending jobs.
    """
    STATUS_PENDING = 'pending'
    STATUS_RUNNING = 'running'
    STATUS_DONE = 'done'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_RUNNING, 'Running'),
        (STATUS_DONE, 'Done'),
        (STATUS_FAILED, 'Failed'),
    ]

    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)
    start_date = models.CharField(max_length=8)  # YYYYMMDD
    end_date = models.CharField(max_length=8)    # YYYYMMDD
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    attempts = models.IntegerField(default=0)
    last_error = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"FetchJob({self.latitude},{self.longitude}) {self.status}"