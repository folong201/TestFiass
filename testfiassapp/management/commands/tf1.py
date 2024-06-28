import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from django.core.management.base import BaseCommand
from testfiassapp.models import ImageFeature

class Command(BaseCommand):
    help = 'Extract features from images and store them in the database'

    def handle(self, *args, **kwargs):
        # Charger le modèle pré-entraîné
        model = models.resnet50(pretrained=True)
        model = model.eval()

        # Transformer pour prétraiter les images
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Fonction pour extraire les caractéristiques
        def extract_features(image_path):
            img = Image.open(image_path)
            img_t = preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                features = model(batch_t)

            return features.numpy().flatten()

        # Chemin du dossier des images
        image_folder = 'images'

        # Boucle pour extraire les caractéristiques et les stocker dans la base de données
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            features = extract_features(image_path)

            # Convertir les caractéristiques en bytes pour les stocker dans la base de données
            features_bytes = features.tobytes()

            # Créer et sauvegarder l'entrée dans la base de données
            image_feature = ImageFeature(image=image_name, features=features_bytes)
            image_feature.save()

            self.stdout.write(self.style.SUCCESS(f'Successfully processed {image_name}'))
