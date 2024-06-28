from django.db import models


class ImageModel:
    def __init__(self):
        # Chargement du modèle VGG16 pré-entraîné
        self.model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    def extract_features(self, image_array):
        """
        Extrait les caractéristiques d'une image sous forme de vecteur.
        
        Args:
            image_array (numpy.ndarray): Tableau numpy représentant l'image.
        
        Returns:
            numpy.ndarray: Vecteur de caractéristiques de l'image.
        """
        # Prétraitement de l'image
        image_array = preprocess_input(image_array)
        image_array = img_to_array(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Extraction des caractéristiques de l'image
        features = self.model.predict(image_array)[0]

        return features

# Create your models here.

# class ImageFeature(models.Model):
#     image_name = models.CharField(max_length=100)
#     features = models.BinaryField()
    

class ImageFeature(models.Model):
    image = models.ImageField(upload_to='kaggle/')
    features = models.BinaryField()
