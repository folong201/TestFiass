# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from django.core.management.base import BaseCommand
# from testfiassapp.models import ImageFeature

# class Command(BaseCommand):
#     help = 'Extract features from images and store them in the database'

#     def handle(self, *args, **kwargs):
#         # Charger le modèle pré-entraîné
#         model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#         # Fonction pour extraire les caractéristiques
#         def extract_features(image_path):
#             img = image.load_img(image_path, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)

#             features = model.predict(img_array)
#             return features.flatten()

#         # Chemin du dossier des images
#         image_folder = 'images'

#         # Boucle pour extraire les caractéristiques et les stocker dans la base de données
#         for image_name in os.listdir(image_folder):
#             image_path = os.path.join(image_folder, image_name)
#             features = extract_features(image_path)

#             # Convertir les caractéristiques en bytes pour les stocker dans la base de données
#             features_bytes = features.tobytes()

#             # Créer et sauvegarder l'entrée dans la base de données
#             image_feature = ImageFeature(image=image_name, features=features_bytes)
#             image_feature.save()

#             self.stdout.write(self.style.SUCCESS(f'Successfully processed {image_name}'))


# testfiassapp/management/commands/extract_features.py
import os
import numpy as np
from django.core.management.base import BaseCommand
from testfiassapp.models import ImageFeature
from testfiassapp.utils import extract_features  # Importer la fonction depuis utils.py

class Command(BaseCommand):
    help = 'Extract features from images and store them in the database'

    def handle(self, *args, **kwargs):
        # Chemin du dossier des images
        image_folder = 'images'  # Utiliser le bon dossier

        # Boucle pour extraire les caractéristiques et les stocker dans la base de données
        for image_name in os.listdir(image_folder):
            # Vérifiez si l'image a déjà été traitée
            if ImageFeature.objects.filter(image=image_name).exists():
                self.stdout.write(self.style.WARNING(f'Skipping {image_name}: already processed'))
                continue
            
            image_path = os.path.join(image_folder, image_name)
            features = extract_features(image_path)

            # Convertir les caractéristiques en bytes pour les stocker dans la base de données
            features_bytes = features.tobytes()

            # Créer et sauvegarder l'entrée dans la base de données
            image_feature = ImageFeature(image=image_name, features=features_bytes)
            image_feature.save()

            self.stdout.write(self.style.SUCCESS(f'Successfully processed {image_name}'))
