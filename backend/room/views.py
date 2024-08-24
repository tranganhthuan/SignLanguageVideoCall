from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Room
from .serializers import RoomSerializer

# Create your views here.
def room(request, room_name, initiator):
    html = f"<html><body><p>This is room {room_name} - {initiator}</p></body></html>"
    return HttpResponse(html)

    
class RoomInfo(APIView):

    def post(self, request):
        room_id = request.data.get('id')
        
        try:
            # Check if the room exists
            room = Room.objects.get(id=room_id)
            # If the room exists, return a message indicating the user has joined
            return Response({"message": "false", "room": RoomSerializer(room).data})
        except Room.DoesNotExist:
            # If the room does not exist, create it
            serializer = RoomSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response({"message": "true", "room": serializer.data})