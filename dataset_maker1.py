import mediapipe as mp
import cv2
import numpy as np
import os

subdir = 'hand_open'  # Alege clasa pentru care să colectezi imagini
n_samples_save = 300  # Numărul de imagini de capturat

# Inițializează  MediaPipe pentru desenare landmark-uri maini
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inițializare  variabile pentru iteratie
iteration_counter = n_samples_save + 1
folder_counter = 1
X, y = [], []

# Maparea categoriilor de gesturi
mapping = {
    'hand_closed': 0,
    'one': 1,
    'two': 2,
    'palm': 4,
    'hand_open': 5,
}

# Inițializează modelul recunoasterea mainilor
hands = mp_hands.Hands(min_detection_confidence=0.3, static_image_mode=True)

# Pornește capturarea video
capture = cv2.VideoCapture(0) #0 pentru integrata\1 pentru cea conectata
while capture.isOpened():
    ret, frame = capture.read()  # Citește un cadru din capturarea video
    frame = cv2.flip(frame, 1)  # Răstoarnă cadrul pe orizontală pentru o vedere oglindită
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertește cadrul în RGB pentru procesarea cu MediaPipe
    detected_image = hands.process(image)  # Procesează cadrul pentru detectarea mainilor
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertește cadrul înapoi în BGR pentru OpenCV

    one_sample = []  # Inițializează o listă pentru a stoca punctele de reper pentru un eșantion

    if detected_image.multi_hand_landmarks:
        # Dacă sunt detectate maini, el le deseneaza pe cadru
        for hand_lms in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_lms,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(255, 0, 255), thickness=4, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(20, 180, 90), thickness=2, circle_radius=2)
            )

   #Afiseaza  cardul
    cv2.imshow('DatasetMaker', image)

    # Verifică dacă s-a apăsat tasta 'r'
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print("Începe capturarea imaginilor")
        iteration_counter = 1

    if iteration_counter < n_samples_save + 1:
        # Daca se apsa tasta r salvează imaginile capturate în subdirectorul specificat
        cv2.imwrite(os.path.join('data', subdir, f'{subdir}_image{iteration_counter}.jpg'), image)
        if detected_image.multi_hand_landmarks:
            # Dacă sunt detectate puncte de reper, le salvează în setul de date
            for hand_lms in detected_image.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    one_sample.extend([lm.x, lm.y])
                X.append(one_sample)
                y.append(mapping[subdir])

        # Afișează un mesaj când toate imaginile pentru categorie sunt salvate
        if iteration_counter == n_samples_save:
            print(f'Imaginile pentru categoria {subdir} au fost salvate.')

        iteration_counter += 1  # Incrementează contorul de iterații

    # Verifică dacă s-a apăsat tasta 'q' pentru exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Opreste capturarea video și închide toate ferestrele OpenCV
capture.release()
cv2.destroyAllWindows()

# Convertește listele în array-uri NumPy și afișează dimensiunea acestora
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# Salvează setul de date într-un fișier .npz
np.savez(os.path.join('data', f'data_{subdir}.npz'), X=X, y=y)
