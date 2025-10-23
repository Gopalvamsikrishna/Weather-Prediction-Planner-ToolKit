# Weather Prediction & Planner ToolKit

A Django-based web application that displays historical weather data.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Will_it_rain_on_my_parade
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    
    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Apply database migrations:**
    ```bash
    python mysite/manage.py migrate
    ```

2.  **Fetch the weather data:**
    This command fetches historical weather data and populates the database.
    ```bash
    python mysite/manage.py fetch_data
    ```

3.  **Run the development server:**
    ```bash
    python mysite/manage.py runserver
    ```

4.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:8000/weather/`.