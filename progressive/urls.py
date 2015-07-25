from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    url(r'^$', 'progressive.views.home', name='home'),
    url(r'^dataset/', include('dataset.urls', namespace='dataset')),
    url(r'^admin/', include(admin.site.urls)),
)
