import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model

def build_multi_output_autoencoder(img_shape=(224, 224, 3), latent_dim=128):
    input_img = Input(shape=img_shape)

    # --- Encodeur ---
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Vecteur Latent
    latent = GlobalAveragePooling2D()(encoded)
    latent = Dense(latent_dim, activation='relu', name='latent_vector')(latent)

    # --- Branche Classification (Supervisée) ---
    class_output = Dense(1, activation='sigmoid', name='class_output')(latent)

    # --- Branche Décodeur (Reconstruction) ---
    x_dec = Dense(14 * 14 * 128, activation='relu')(latent)
    x_dec = Reshape((14, 14, 128))(x_dec)
    x_dec = UpSampling2D((2, 2))(x_dec)
    reconstruction = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='reconstruction')(x_dec)

    model = Model(inputs=input_img, outputs=[class_output, reconstruction])
    return model
