import os
import numpy as np
import faiss
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from testfiassapp.utils import extract_features  # Assurez-vous que cette fonction est correctement importée
from .models import ImageFeature

def index(request):
    return render(request, 'upload.html')

def upload_and_search(request):
    if request.method == 'POST' and 'imageup' in request.FILES:
        # Enregistrer l'image téléchargée
        uploaded_image = request.FILES['imageup']
        image_path = default_storage.save('uploads/' + uploaded_image.name, uploaded_image)

        # Extraire les caractéristiques de l'image téléchargée
        features = extract_features(os.path.join(settings.MEDIA_ROOT, image_path))

        # Charger les caractéristiques de la base de données
        image_features = ImageFeature.objects.all()
        db_features = np.array([np.frombuffer(image_feature.features, dtype=np.float32) for image_feature in image_features])
        image_paths = [os.path.join(settings.MEDIA_URL, 'images', image_feature.image.name) for image_feature in image_features]

        # Créer l'index FAISS et ajouter les caractéristiques
        index = faiss.IndexFlatL2(features.shape[0])
        index.add(db_features)

        # Rechercher les images similaires
        _, indices = index.search(np.expand_dims(features, axis=0), k=5)
        similar_images = [image_features[int(i)] for i in indices[0]]  # Convertir les indices en int

        # Afficher les propriétés des images similaires dans la console
        for img in similar_images:
            print(f'Image Name: {img.image.name}, Image Path: {os.path.join(settings.MEDIA_URL, img.image.name)}')

        # Passer les informations au template
        similar_images_info = [{'name': img.image.name, 'url': os.path.join(settings.MEDIA_URL, img.image.name)} for img in similar_images]

        return render(request,'result.html', {'images': similar_images_info})

    return render(request, 'upload.html')

