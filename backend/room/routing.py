from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/webcam/$', consumers.WebcamConsumer.as_asgi()),
    re_path(r'ws/sign/(?P<room_name>\w+)/(?P<initiator>true|false)$', consumers.SignDetectionConsumer.as_asgi()),
    re_path(r'ws/video/(?P<room_name>\w+)/$', consumers.VideoConsumer.as_asgi()),
]