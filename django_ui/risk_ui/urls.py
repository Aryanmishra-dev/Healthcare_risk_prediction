"""
URL configuration for risk_ui.
"""

from django.urls import path
from risk_ui import views

urlpatterns = [
    path("", views.predict_view, name="predict"),
]
