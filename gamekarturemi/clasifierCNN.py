import os
from keras.models import load_model
import cv2
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
#from keras.utils import img_to_array

# Fungsi untuk memuat data pelatihan dari semua label dalam folder CardDataSet
def LoadCitraTraining(sDir):
    label_folders = [d for d in os.listdir(sDir) if os.path.isdir(os.path.join(sDir, d))]
    num_classes = len(label_folders)
    target_classes = np.eye(num_classes)
    X = []  # Menampung data citra
    T = []  # Menampung target

    for i, label in enumerate(label_folders):
        label_dir = os.path.join(sDir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(file_path)
                img = cv2.resize(img, (128, 128))
                img = img.astype('float32') / 255.0
                X.append(img)
                T.append(target_classes[i])

    X = np.array(X)
    T = np.array(T)
    return X, T, label_folders

# Membuat model CNN sederhana
def ModelDeepLearningCNN(num_classes):
    input_img = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_img, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Fungsi untuk melatih model dengan data dari CardDataSet
def TrainingCNN(epochs, dataset_dir, model_save_path):
    X, Y, label_folders = LoadCitraTraining(dataset_dir)
    num_classes = len(label_folders)
    model = ModelDeepLearningCNN(num_classes)
    history = model.fit(X, Y, epochs=epochs, shuffle=True, batch_size=32)
    model.save(model_save_path)
    print(f"Training completed. Model saved to {model_save_path}")

# Fungsi Klasifikasi untuk prediksi gambar
def Klasifikasi(model_path, image_path, label_folders):
    # Memuat model yang sudah dilatih
    model = load_model(model_path)
    
    # Preprocessing gambar input
    if not os.path.exists(image_path):
        print(f"Error: Gambar tidak ditemukan di {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gambar gagal dimuat dari {image_path}")
        return None
    
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Membuat bentuknya (1, 128, 128, 3)

    # Prediksi dengan model
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    predicted_label = label_folders[predicted_index]
    
    return predicted_label

# Pengaturan pelatihan
dataset_dir = "CardDataSet"  # Lokasi dataset
model_save_path = "card_classification_model.h5"  # Lokasi penyimpanan model
TrainingCNN(epochs=10, dataset_dir=dataset_dir, model_save_path=model_save_path)
