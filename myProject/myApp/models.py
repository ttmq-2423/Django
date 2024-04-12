from django.db import models

class myApp(models.Model):
    predicted_class_name = models.CharField(max_length=50)

    def __str__(self):
        return self.predicted_class_name    
