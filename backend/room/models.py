from django.db import models

class Room(models.Model):
    id = models.CharField(max_length=50, primary_key=True)

    def __str__(self):
        return f"Room {self.id}"