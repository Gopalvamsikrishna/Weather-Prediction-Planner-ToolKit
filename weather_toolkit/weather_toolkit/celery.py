from __future__ import annotations

import os
from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_toolkit.settings')

app = Celery('weather_toolkit')

# Using a string here means the worker will not have to
# pickle the object when using Windows/Heroku; see Celery docs.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
