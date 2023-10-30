from django.urls import path, include

urlpatterns = [
    path('', include('RecommendationSystem_App.urls.Dashboard.Home_urls')),
]