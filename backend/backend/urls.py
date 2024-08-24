from django.urls import path
from room import views
from django.urls import register_converter
from . import converters

register_converter(converters.BooleanConverter, 'bool')

urlpatterns = [
    path('room/<str:room_name>/<bool:initiator>/', views.room),
    path('room_info/', views.RoomInfo.as_view()),
    
]
