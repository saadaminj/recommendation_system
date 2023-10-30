from django.urls import path, include
from RecommendationSystem_Admin.views.Auth import auth_views

urlpatterns = [
    path('',auth_views.login, name = 'Login'),
    path('SignUp/',auth_views.signUp,name='signUp'),
    path('ForgetPassword/',auth_views.forgetPassword,name='forgetPassword'),
    path('Logout',auth_views.logout_view,name='logout_view'),
]