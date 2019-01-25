"""djangosample URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from api import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('searchengine/', include('searchengine.urls')),
    url(r'google/',views.getGoogle),
    url(r'bing/',views.getBing),
    url(r'ourSearchEngine/',views.ourSearchEngine),
    url(r'getAll/', views.getAll),
    url(r'getPageRank/', views.getPageRankResults),
    url(r'getHitResults/', views.getHitResults),
    url(r'getRocchioResults/', views.getRocchioresults),
    url(r'getMetricResults/', views.getMetricresults),
    url(r'getkmeansResults/', views.getKMeansresults),
    url(r'getAggResults/', views.getAggresults),

    #getAggresults
]
