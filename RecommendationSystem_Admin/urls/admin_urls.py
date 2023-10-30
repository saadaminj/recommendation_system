from django.urls import path, include
urlpatterns = [
    path('',include('RecommendationSystem_Admin.urls.Auth.Auth_urls')),
    path('Dashboard/',include('RecommendationSystem_Admin.urls.Dashboard.AdminDashboard_urls')),
    path('Product/',include('RecommendationSystem_Admin.urls.Products.product_urls')),
    path('Category/',include('RecommendationSystem_Admin.urls.Category.Category_urls')),
    path('Subcategory/',include('RecommendationSystem_Admin.urls.SubCategory.SubCategory_urls')),
    path('AppUser/',include('RecommendationSystem_Admin.urls.AppUser.appUser_urls')),
    path('AdminUser/',include('RecommendationSystem_Admin.urls.AdminUser.adminUser_urls')),
]