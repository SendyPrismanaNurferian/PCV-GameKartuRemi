import cv2
import numpy as np
from keras.models import load_model
from clasifierCNN import Klasifikasi, LoadCitraTraining

# Load model yang telah dilatih
model = load_model("card_classification_model.h5")

# Load daftar label dari dataset
_, _, label_folders = LoadCitraTraining("CardDataSet")

# Fungsi untuk menangkap kartu dari kamera dan memprediksi
def detect_card_from_camera():
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default (index 0)

    if not cap.isOpened():
        print("Gagal membuka kamera.")
        return

    print("Tekan [ESC] untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Deteksi dan warp kartu dari frame kamera
        processed_card = detect_and_warp_cards(frame)
        if processed_card is not None:
            # Simpan gambar sementara untuk prediksi
            temp_image_path = "temp_card.jpg"
            cv2.imwrite(temp_image_path, processed_card)

            # Prediksi label kartu
            predicted_label = Klasifikasi("card_classification_model.h5", temp_image_path, label_folders)
            if predicted_label:
                print(f"Kartu terdeteksi: {predicted_label}")
            
            # Tampilkan hasil prediksi di jendela video
            cv2.putText(frame, f"Kartu: {predicted_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Warped Card", processed_card)
        else:
            cv2.putText(frame, "Tidak ada kartu terdeteksi", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Tampilkan frame dari kamera
        cv2.imshow('Live Camera Feed', frame)

        # Tekan ESC untuk keluar
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi deteksi dan warp kartu (deteksi dari frame kamera)
def detect_and_warp_cards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Hanya deteksi kontur yang lebih besar dari 1000 px
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:  # Cari kontur dengan 4 titik (persegi panjang/kartu)
                pts = sorted(approx, key=lambda x: x[0][1])
                top_points = sorted(pts[:2], key=lambda x: x[0][0])
                bottom_points = sorted(pts[2:], key=lambda x: x[0][0])

                bmin, bmax = top_points[0][0], top_points[1][0]
                kmin, kmax = bottom_points[0][0], bottom_points[1][0]
                pts_array = np.array([bmin, bmax, kmax, kmin], dtype="float32")
                pts_array = force_vertical(pts_array)

                # Ukuran tetap yang diinginkan untuk gambar hasil warp
                fixed_width, fixed_height = 200, 300
                destination = np.array([[0, 0], [fixed_width - 1, 0], 
                                        [fixed_width - 1, fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(pts_array, destination)
                warped = cv2.warpPerspective(frame, M, (fixed_width, fixed_height))
                return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    return None

# Fungsi untuk menegakkan kartu (agar orientasinya benar)
def force_vertical(box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    if width > height:
        box = np.roll(box, 1, axis=0)
    return box

# Panggil fungsi untuk mendeteksi kartu dari kamera
detect_card_from_camera()