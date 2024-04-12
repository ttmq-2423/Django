from rest_framework import serializers

from myApp.models import myApp

class myAppSerializer(serializers.ModelSerializer):

    class Meta: 
        model = myApp
        fields = ["predicted_class_name"]
