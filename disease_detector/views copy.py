from django.shortcuts import render
from .models import UploadedImage
from .predict import prediction_main


# Create your views here
def predict_disease(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        uploaded_image = UploadedImage.objects.create(image=image)
        prediction_accuracy, predicted_class = prediction_main(uploaded_image.image.path)
        image_path = uploaded_image.image.url
        return render(request, 'result.html', {'predicted_class': predicted_class, 'prediction_accuracy': prediction_accuracy, 'image_path': image_path})
    return render(request, 'index.html')


def camera_capture(request):
    return render(request, 'camera_capture.html')
