from django.shortcuts import render
from urllib.request import urlopen
from django.core.files import File
from django.core.files.temp import NamedTemporaryFile
from .models import Image
from .predict import prediction_main


# Create your views here
def predict_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        uploaded_image = Image.objects.create(image=image)
        prediction_accuracy, predicted_class, description = prediction_main(uploaded_image.image.path)
        image_path = uploaded_image.image.url
        return render(request, 'result.html', {'predicted_class': predicted_class, 'prediction_accuracy': prediction_accuracy, 'image_path': image_path, 'description' : description})
    return render(request, 'index.html')


def camera_capture(request):
    if request.method == 'POST':
        path = request.POST["src"]
        image = NamedTemporaryFile()
        image.write(urlopen(path).read())
        image.flush()
        image = File(image)
        name = str(image.name).split('\\')[-1]
        name += '.jpg'
        image.name = name
        obj = Image.objects.create(image=image)
        obj.save()
        
        # Perform prediction on the submitted image
        prediction_accuracy, predicted_class, description = prediction_main(obj.image.path)
        
        # Redirect to the result page with prediction details
        return render(request, 'result.html', {'predicted_class': predicted_class, 'prediction_accuracy': prediction_accuracy, 'image_path': obj.image.url, 'description': description})
    return render(request, 'camera_capture.html')

