from django.urls import path, include
from RecommendationSystem_Admin.views.AdminUser import adminUser_views

urlpatterns = [
    path('',adminUser_views.adminUserList,name='adminUserList'),
    path('New',adminUser_views.addNewAdminUser,name='addNewAdminUser'),
    path('Details/<int:id>',adminUser_views.adminUserInfo,name='adminUserInfo'),
    path('Edit/<int:id>',adminUser_views.editAdminUserDetails,name='editAdminUserDetails'),
    path('Delete/<int:id>',adminUser_views.deleteAdminUser,name='deleteAdminUser'),
]
