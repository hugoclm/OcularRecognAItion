### README.md 

# AI for Ocular Pathology Detection: A Deep Learning Approach

## At a Glance

This repository showcases my technical contribution to the **Ocular RecognAItion** project developed at ESIEE Paris. The objective was to design a diagnostic support system capable of identifying major eye diseases—**Glaucoma, AMD, and Diabetic Retinopathy**—from retinal fundus images using advanced Deep Learning architectures.

> **Note:** This repository is a portfolio intended to present AI research and logic. The full application is no longer live, and model weights are not provided for privacy reasons.

-----

## 🧠 AI Strategy & Architecture

The project was divided into two critical AI stages: **Quality Assessment** and **Automated Diagnosis**.

### 1\. Image Quality Control (EfficientNet)

To ensure diagnostic reliability, I worked on a classification model that filters out unexploitable images (blur, poor lighting, or missing optic nerve).

  * **Architecture:** EfficientNet-B0 (Transfer Learning).
  * **Performance:** Reached **95% accuracy** in distinguishing "Good" vs "Bad" quality images.

### 2\. Diagnostic Engine (ResNet-50 & Autoencoders)

The core logic utilizes a two-step training process to handle the complexity of medical imaging.

  * **Phase 1:** An **Autoencoder** was trained to learn robust visual representations of the retina (Loss \~0.001).
  * **Phase 2:** The pre-trained encoder was integrated into a **ResNet-50** classifier.
  * **Results:**
      * **Multiclass Accuracy:** 87%.
      * **Binary Accuracy (AMD & DR):** \~95%.
      * **Binary Accuracy (Glaucoma):** 60-65% (identified as a key area for future optimization).

-----

## 📊 Visual Insights

### Model Performance

Below is a confusion matrix illustrating the model's ability to classify pathologies accurately during the validation phase.

> !(assets/confusionMatrix)

### Interface Design

The tool featured a professional dashboard allowing doctors to upload images or capture them via a custom 3D-printed adapter for iPhone 15.

> *\[Insert your dashboard\_screenshot.png here\]*

-----

## 🛠️ Tech Stack & Key Snippets

  * **Frameworks:** PyTorch (Classification), TensorFlow/Keras (Quality Control).
  * **Processing:** OpenCV, NumPy, Pillow.
  * **Backend:** Flask (for AI model serving logic).

**Sample Logic (Model Definition):**

``` python
# Extract from model_def.py
# Highlighting the custom head for medical image classification
self.classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes) # num_classes = 4 (Healthy, Glaucoma, AMD, DR)
)

```

-----

## 📝 Key Learning & Perspectives

This project highlights the challenges of **medical data scarcity** and the importance of **data augmentation** and **autoencoders** in improving model convergence. Future work would focus on improving Glaucoma detection through more granular segmentation of the optic cup-to-disc ratio.

-----

Souhaitez-vous que je vous aide à rédiger une section plus détaillée sur la partie "Autoencodeur", qui est techniquement très valorisante pour votre portfolio ?
