from django.shortcuts import render
from django.http import JsonResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from rest_framework.decorators import api_view
from .models import myApp
from .serializer import myAppSerializer

class_names = ['daisy', 'dandelion'] 

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to match the original model's output units
state_dict = torch.load('flower_classification_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

new_model = models.resnet18(pretrained=True)
new_model.fc = nn.Linear(new_model.fc.in_features, 2) 

new_model.fc.weight.data = model.fc.weight.data[0:2]  
new_model.fc.bias.data = model.fc.bias.data[0:2]


@api_view(['POST'])
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('imageFile'):
        image_file = request.FILES['imageFile']
        image = Image.open(image_file)
        image.save('myApp/static/myApp/' +  image_file.name)  
        result = processed_img('myApp/static/myApp/' + image_file.name)
        my_model = myApp.objects.create(predicted_class_name = result)
        my_model.save()

        serializer = myAppSerializer(my_model)
        image.close()
        return JsonResponse(serializer.data, status = 200)
    else:
        return JsonResponse({'error': 'No image file found.'})



def processed_img(img_path):

    image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_class = output.max(1)
    predicted_class_name = class_names[predicted_class.item()]  
    return predicted_class_name

