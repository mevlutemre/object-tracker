import cv2
import numpy as np
import serial
import time

try:
    ser = serial.Serial('COM6', 57600, timeout=0.1) 
    seri_acik = True
except:
    print("Seri port açılamadı! Komutlar sadece ekrana yazdırılacak.")
    seri_acik = False

def servo_sur(yaw_aci, pitch_aci):
    komut = f"{int(yaw_aci):03d}-{int(pitch_aci):03d}\n"
    if seri_acik:
        ser.write(komut.encode())
    return komut

kamera = cv2.VideoCapture(0)

cam_genislik = int(kamera.get(3))
cam_yukseklik = int(kamera.get(4))

yaw = 90.0
pitch = 90.0
yaw_yon = 1     
pitch_yon = 1

tarama_hizi_yaw = 3.0
tarama_hizi_pitch = 10.0

son_komut_zamani = time.time()

# --- YENİ DEĞİŞKENLER (ROI VE KİLİTLENME İÇİN) ---
kilitli = False
roi_kutu = None  # [x, y, w, h] formatında ana ekrandaki ROI koordinatları
roi_genisletme = 40  # Nesnenin kaç piksel etrafını tarayalım? (Tolerans payı)

while True:
    basarili_mi, frame = kamera.read()
    if not basarili_mi:
        break
        
    frame = cv2.flip(frame, 1)
    
    # --- ROI KESME İŞLEMİ ---
    # Eğer kilitliysek ve geçerli bir ROI varsa, resmi o bölgeye kırpıyoruz
    if kilitli and roi_kutu is not None:
        rx, ry, rw, rh = roi_kutu
        # Görüntü sınırlarının dışına çıkmamak için önlem alıyoruz
        rx1 = max(0, rx - roi_genisletme)
        ry1 = max(0, ry - roi_genisletme)
        rx2 = min(cam_genislik, rx + rw + roi_genisletme)
        ry2 = min(cam_yukseklik, ry + rh + roi_genisletme)
        
        isleme_alani = frame[ry1:ry2, rx1:rx2]
        # Ekranda ROI alanını görmek için mavi bir kutu çizelim
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
    else:
        # Kilitli değilsek tüm ekranı işle
        isleme_alani = frame
        rx1, ry1 = 0, 0 # Koordinat dönüşümü için ofset sıfır

    # Renk tespiti artık tüm ekranda değil, sadece belirlenen 'isleme_alani' içinde yapılıyor
    hsv = cv2.cvtColor(isleme_alani, cv2.COLOR_BGR2HSV)

    maske1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    maske2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    tam_maske = maske1 + maske2
    tam_maske = cv2.erode(tam_maske, None, iterations=2)
    tam_maske = cv2.dilate(tam_maske, None, iterations=2)

    konturlar, _ = cv2.findContours(tam_maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    nesne_bulundu = False

    if len(konturlar) > 0:
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
        
        if cv2.contourArea(en_buyuk_kontur) > 500:
            nesne_bulundu = True
            kilitli = True # Nesne şartları sağlıyorsa kilidi aktif et
            
            # Kırpılmış alandaki yerel (local) koordinatlar
            lx, ly, lw, lh = cv2.boundingRect(en_buyuk_kontur)
            
            # Yerel koordinatları ana ekran (global) koordinatlarına çeviriyoruz
            x = lx + rx1
            y = ly + ry1
            w = lw
            h = lh
            
            # Bir sonraki frame'de kullanılmak üzere ROI kutusunu güncelle
            roi_kutu = [x, y, w, h]
            
            # Nesneyi yeşil kutuya al
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "LOCKED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            merkez_x = x + w / 2
            merkez_y = y + h / 2
            
            hata_x = merkez_x - (cam_genislik / 2)
            hata_y = merkez_y - (cam_yukseklik / 2)
            
            yaw += hata_x * 0.01   
            pitch -= hata_y * 0.01 

    # Eğer nesne bulunamadıysa kilidi kır ve radar taramasına geri dön
    if not nesne_bulundu:
        kilitli = False
        roi_kutu = None
        
        yaw += tarama_hizi_yaw * yaw_yon
        
        if yaw >= 150:
            yaw = 150
            yaw_yon = -1
            pitch += tarama_hizi_pitch * pitch_yon
        elif yaw <= 30:
            yaw = 30
            yaw_yon = 1
            pitch += tarama_hizi_pitch * pitch_yon
            
        if pitch >= 150:
            pitch = 150
            pitch_yon = -1
        elif pitch <= 30:
            pitch = 30
            pitch_yon = 1

    yaw = max(30, min(150, yaw))
    pitch = max(30, min(150, pitch))

    su_an = time.time()
    if su_an - son_komut_zamani > 0.05:
        giden_komut = servo_sur(yaw, pitch)
        cv2.putText(frame, f"Giden Komut: {giden_komut.strip()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        son_komut_zamani = su_an

    cv2.imshow("Tracker Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
if seri_acik:
    ser.close()