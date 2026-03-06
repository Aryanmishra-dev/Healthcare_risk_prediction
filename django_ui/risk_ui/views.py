"""
Views for the diabetes risk prediction UI.
"""

import requests
from django.shortcuts import render

FASTAPI_URL = "http://127.0.0.1:8000/predict"


def predict_view(request):
    """Handle the prediction form and display results."""
    context = {"result": None, "error": None}

    if request.method == "POST":
        try:
            payload = {
                "age": float(request.POST.get("age", 7)),
                "bmi": float(request.POST.get("bmi", 25)),
                "bp": float(request.POST.get("bp", 0)),
                "cholesterol": float(request.POST.get("cholesterol", 0)),
                "smoker": float(request.POST.get("smoker", 0)),
                "activity": float(request.POST.get("activity", 1)),
                "health": float(request.POST.get("health", 3)),
                "mental": float(request.POST.get("mental", 0)),
            }
            response = requests.post(FASTAPI_URL, json=payload, timeout=10)
            response.raise_for_status()
            context["result"] = response.json()
            context["form_data"] = payload
        except requests.exceptions.ConnectionError:
            context["error"] = "Cannot connect to the prediction API. Make sure the FastAPI server is running on port 8000."
        except requests.exceptions.RequestException as e:
            context["error"] = f"API request failed: {e}"
        except (ValueError, TypeError) as e:
            context["error"] = f"Invalid input: {e}"

    return render(request, "predict.html", context)
