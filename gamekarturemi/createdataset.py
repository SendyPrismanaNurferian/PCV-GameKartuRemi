import cv2
import os
import datetime
import time
import numpy as np
import clasifierCNN as csNNN

# Daftar label kartu sesuai dengan penamaan yang diinginkan
card_labels = [
    "ace_of_spades", "two_of_spades", "three_of_spades", "four_of_spades", "five_of_spades", "six_of_spades",
    "seven_of_spades", "eight_of_spades", "nine_of_spades", "ten_of_spades", "jack_of_spades", "queen_of_spades",
    "king_of_spades", "ace_of_hearts", "two_of_hearts", "three_of_hearts", "four_of_hearts", "five_of_hearts",
    "six_of_hearts", "seven_of_hearts", "eight_of_hearts", "nine_of_hearts", "ten_of_hearts", "jack_of_hearts",
    "queen_of_hearts", "king_of_hearts", "ace_of_clubs", "two_of_clubs", "three_of_clubs", "four_of_clubs",
    "five_of_clubs", "six_of_clubs", "seven_of_clubs", "eight_of_clubs", "nine_of_clubs", "ten_of_clubs",
    "jack_of_clubs", "queen_of_clubs", "king_of_clubs", "ace_of_diamonds", "two_of_diamonds", "three_of_diamonds",
    "four_of_diamonds", "five_of_diamonds", "six_of_diamonds", "seven_of_diamonds", "eight_of_diamonds",
    "nine_of_diamonds", "ten_of_diamonds", "jack_of_diamonds", "queen_of_diamonds", "king_of_diamonds"
]

# Fungsi untuk membuat direktori jika belum ada
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Fungsi untuk mendapatkan nama file berdasarkan waktu
def get_file_name():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S%f')

# Fungsi untuk menegakkan kartu
def force_vertical(box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    if width > height:
        box = np.roll(box, 1, axis=0)
    return box

# Fungsi deteksi dan warp kartu
def detect_and_warp_cards(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = sorted(approx, key=lambda x: x[0][1])
                top_points = sorted(pts[:2], key=lambda x: x[0][0])
                bottom_points = sorted(pts[2:], key=lambda x: x[0][0])

                bmin, bmax = top_points[0][0], top_points[1][0]
                kmin, kmax = bottom_points[0][0], bottom_points[1][0]
                pts_array = np.array([bmin, bmax, kmax, kmin], dtype="float32")
                pts_array = force_vertical(pts_array)

                fixed_width, fixed_height = 200, 300
                destination = np.array([[0, 0], [fixed_width - 1, 0], 
                                        [fixed_width - 1, fixed_height - 1], [0, fixed_height - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(pts_array, destination)
                warped = cv2.warpPerspective(frame, M, (fixed_width, fixed_height))
                return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    return None

# Fungsi utama untuk mengumpulkan dataset
def collect_dataset(output_dir, camera_index=0, capture_duration=10, images_per_second=15):
    cap = cv2.VideoCapture(camera_index)
    label_index = 0
    is_saving = False
    start_time = None
    total_images_captured = 0

    print("=== KONTROL PENGAMBILAN DATASET ===")
    print("Tekan [SPASI] untuk mulai/menghentikan pengambilan gambar")
    print("Tekan [N] untuk mengganti kartu berikutnya")
    print("Tekan [R] untuk reset ke kartu pertama")
    print("Tekan [ESC] untuk keluar program")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal menangkap frame.")
            break

        cv2.putText(
            frame, f"Kartu: {card_labels[label_index]}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        if is_saving:
            elapsed_time = time.time() - start_time
            if elapsed_time <= capture_duration:
                if total_images_captured < capture_duration * images_per_second:
                    processed_card = detect_and_warp_cards(frame)
                    if processed_card is not None:
                        class_dir = os.path.join(output_dir, card_labels[label_index])
                        create_directory(class_dir)
                        file_name = os.path.join(class_dir, get_file_name() + ".jpg")
                        cv2.imwrite(file_name, processed_card)
                        total_images_captured += 1
                        print(f"Captured image {total_images_captured} for {card_labels[label_index]}")

                        # Tampilkan hasil kartu yang telah diluruskan dan difilter grayscale
                        cv2.imshow("Warped Grayscale Card", processed_card)
            else:
                print(f"Pengambilan gambar selesai untuk kartu {card_labels[label_index]}")
                is_saving = False

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  
            is_saving = not is_saving
            start_time = time.time()
            total_images_captured = 0
            if is_saving:
                print(f"Mulai merekam untuk kartu {card_labels[label_index]}")
            else:
                print(f"Rekam selesai untuk kartu {card_labels[label_index]}")
        elif key == ord('n'):  
            label_index = (label_index + 1) % len(card_labels)
            print(f"Beralih ke kartu {card_labels[label_index]}")
        elif key == ord('r'):
            label_index = 0
            print("Reset ke kartu pertama")
        elif key == 27:  # Tekan ESC untuk keluar
            print("Program selesai, selamat anda keluar")
            break

    cap.release()
    cv2.destroyAllWindows()

# Panggil fungsi untuk mengumpulkan dataset
output_dir = "CardDataSet"
collect_dataset(output_dir)
