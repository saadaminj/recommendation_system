from django.urls import path, include
from RecommendationSystem_Admin.views.Products import product_views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('List',product_views.productDetails,name='productDetails'),
    path('Addnew',product_views.addNewProduct,name='addNewProduct'),
    path('Info/<int:id>',product_views.productInfo,name='productInfo'),
    path('Edit/<int:id>',product_views.editProductDetails,name='editProductDetails'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)