from django.urls import path
from RecommendationSystem_App.views import dashboard_views

urlpatterns = [
    path('',dashboard_views.HomeView,name='HomePage'),
    path('Login', dashboard_views.Applogin,name='Applogin'),
    path('SignUp', dashboard_views.AppSignUp,name='AppSignup'),
    path('Logout', dashboard_views.appLogout,name='appLogout'),
]