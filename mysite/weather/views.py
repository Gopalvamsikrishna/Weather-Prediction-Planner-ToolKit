from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

# --- Template View (for the main interface) ---
def index(request):
    """Renders the main application interface (the map and controls)."""
    # The template must be placed in: weather/templates/weather/index.html
    return render(request, 'weather/index.html')

# --- API Endpoint Placeholders (to satisfy urls.py) ---
# These functions will later be filled with the analytical Python logic.

def percentiles(request):
    """API endpoint to get percentiles. Currently a placeholder."""
    # Placeholder response
    return JsonResponse({"status": "placeholder", "endpoint": "percentiles", "method": request.method})

def probability_view(request):
    """API endpoint to get probability. Currently a placeholder."""
    # Placeholder response
    return JsonResponse({"status": "placeholder", "endpoint": "probability", "method": request.method})

def trend_view(request):
    """API endpoint to get trend data. Currently a placeholder."""
    # Placeholder response
    return JsonResponse({"status": "placeholder", "endpoint": "trend", "method": request.method})

def history_view(request):
    """API endpoint to get historical values. Currently a placeholder."""
    # Placeholder response
    return JsonResponse({"status": "placeholder", "endpoint": "history", "method": request.method})