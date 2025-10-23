from django.db import models

class HistoricalDataPoint(models.Model):
    """
    Stores historical daily weather data retrieved from NASA POWER API.
    The data is indexed by latitude, longitude, and date.
    """
    # 1. Geospatial and Date Keys (Primary Index)
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)
    date = models.DateField(db_index=True, unique=True) # Ensures unique entry per day/location combination

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