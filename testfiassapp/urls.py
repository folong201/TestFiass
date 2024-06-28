from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("searchimage", views.upload_and_search, name="searchimage"),
]
