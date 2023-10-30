from django.urls import path, include
from RecommendationSystem_Admin.views.Category import Category_views

urlpatterns = [
    path('',Category_views.CategoryList,name='CategoryList'),        
    path('AddNew',Category_views.addNewCategory,name='addNewCategory'),
    path('Edit/<int:id>',Category_views.editCategoryDetails,name='editCategoryDetails'),
    path('Delete/<int:id>',Category_views.deleteCategory,name='deleteCategory'),
]
