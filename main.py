import cv2
import numpy as np

kamera = cv2.VideoCapture(0)

while True:
    basarili_mi, frame = kamera.read()
    
    if not basarili_mi:
        print("Kameradan görüntü alınamadı!")
        break
        
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    alt_kirmizi1 = np.array([0, 120, 70])
    ust_kirmizi1 = np.array([10, 255, 255])
    maske1 = cv2.inRange(hsv, alt_kirmizi1, ust_kirmizi1)

    alt_kirmizi2 = np.array([170, 120, 70])
    ust_kirmizi2 = np.array([180, 255, 255])
    maske2 = cv2.inRange(hsv, alt_kirmizi2, ust_kirmizi2)

    tam_maske = maske1 + maske2

    tam_maske = cv2.erode(tam_maske, None, iterations=2)
    tam_maske = cv2.dilate(tam_maske, None, iterations=2)

    konturlar, _ = cv2.findContours(tam_maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(konturlar) > 0:
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
        
        if cv2.contourArea(en_buyuk_kontur) > 500:
            x, y, w, h = cv2.boundingRect(en_buyuk_kontur)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            merkez_x = int(x + w/2)
            merkez_y = int(y + h/2)
            
            print(f"Kırmızı Nesne Takip Ediliyor -> X: {merkez_x}, Y: {merkez_y}")

    cv2.imshow("Kirmizi Nesne Takibi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()