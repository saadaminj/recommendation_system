from django.urls import path, include
from RecommendationSystem_Admin.views.AppUser import appUser_views

urlpatterns = [
    path('',appUser_views.appUserList,name='appUserList'),
    path('New',appUser_views.addNewAppUser,name='addNewAppUser'),
    path('Details/<int:id>',appUser_views.appUserInfo,name='appUserInfo'),
    path('Edit/<int:id>',appUser_views.editAppUserDetails,name='editAppUserDetails'),
    path('Delete/<int:id>',appUser_views.deleteAppUser,name='deleteAppUser'),
]
