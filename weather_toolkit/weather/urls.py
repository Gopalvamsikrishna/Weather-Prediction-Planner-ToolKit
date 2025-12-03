from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('percentiles', views.percentiles, name='percentiles'),
    path('probability', views.probability_view, name='probability'),
    path('trend', views.trend_view, name='trend'),
    path('history', views.history_view, name='history'),
    path('fetch-job-status', views.fetch_job_status, name='fetch-job-status'),
]
