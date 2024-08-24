from channels.generic.websocket import AsyncWebsocketConsumer
import json
import logging

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')
            message_content = text_data_json.get('message')

            if not message_type or not message_content:
                raise ValueError("Invalid message format")

            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'send_message',
                    'message_type': message_type,
                    'message_content': message_content
                }
            )
        except json.JSONDecodeError:
            logging.error("Received invalid JSON")
        except ValueError as e:
            logging.error(f"Received invalid message: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing message: {str(e)}")

    # Receive message from room group
    async def send_message(self, event):
        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': event['message_type'],
            'message': event['message_content']
        }))


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': 'Successfully connected to the room'
        }))

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_message',
                'message': message
            }
        )

    # Receive message from room group
    async def send_message(self, event):
        message = event['message']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'type': event['type'],
            'message': message
        }))