import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from model_def import ResNetClassifier

# Mapping des pathologies détectées
class_names = ['ARMD', 'Glaucome']

def load_model(weights_path):
    # Initialisation de l'architecture sans poids pré-entraînés pour charger les nôtres
    resnet = models.resnet50(weights=None)
    encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
    model = ResNetClassifier(encoder=encoder, num_classes=len(class_names))
    
    # Chargement des poids du modèle entraîné
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    
    return class_names[predicted]
