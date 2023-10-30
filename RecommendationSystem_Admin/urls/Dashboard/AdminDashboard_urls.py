from django.urls import path, include
from RecommendationSystem_Admin.views.Dashboard import adminDashboard_views

urlpatterns = [
    path('',adminDashboard_views.AdminDashboardView,name='AdminDashboardView'),
]