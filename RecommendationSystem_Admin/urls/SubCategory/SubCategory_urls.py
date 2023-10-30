from django.urls import path, include
from RecommendationSystem_Admin.views.SubCategory import subcategory_views

urlpatterns = [
    path('',subcategory_views.SubCategoryList,name='SubCategoryList'),
    path('AddNew',subcategory_views.addNewSubCategoryList,name='addNewSubCategoryList'),
    path('Edit/<int:id>',subcategory_views.editSubCategoryList,name='editSubCategoryList'),
    path('Details/<int:id>',subcategory_views.subcategoryInfo,name='subcategoryInfo'),
    path('Delete/<int:id>',subcategory_views.deleteSubcategory,name='deleteSubcategory'),
]
