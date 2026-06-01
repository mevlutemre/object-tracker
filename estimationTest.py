import cv2
import numpy as np
import serial
import time

try:
    ser = serial.Serial('COM7', 57600, timeout=0.1) 
    seri_acik = True
except:
    print("Seri port açılamadı! Komutlar sadece ekrana yazdırılacak.")
    seri_acik = False

def servo_sur(yaw_aci, pitch_aci):
    komut = f"{int(yaw_aci):03d}-{int(pitch_aci):03d}\n"
    if seri_acik:
        ser.write(komut.encode())
    return komut

# CAP_DSHOW ile hızlı açılış
kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cam_genislik = 640
cam_yukseklik = 480

yaw = 90.0
pitch = 90.0
yaw_yon = 1     
pitch_yon = 1

tarama_hizi_yaw = 3.0
tarama_hizi_pitch = 10.0
son_komut_zamani = time.time()

# --- ROI VE KİLİTLENME DEĞİŞKENLERİ ---
kilitli = False
kilit_onaylandi = False
kilit_suresi_sayaci = 0
gerekli_onay_frame = 45  
roi_kutu = None  
roi_genisletme = 60  

# --- ESTIMATION (TAHMİN) DEĞİŞKENLERİ ---
onceki_merkez_x = None
onceki_merkez_y = None
hiz_x = 0.0  
hiz_y = 0.0  
tahmin_modu = False
tahmin_sayaci = 0
maks_tahmin_frame = 40  
tahmin_hiz_carpani = 1.8  

# --- PID KONTROLCÜ PARAMETRELERİ ---
# İstediğin değerleri doğrudan buraya tanımladım Mevlüt
kp_x = 0.028
kp_y = 0.028

kd_x = 0.008
kd_y = 0.008

ki_x = 0.001  # İntegral kazancı (Gerekirse 0.001 gibi ufak değerlerle başlatabilirsin)
ki_y = 0.001

# PID Hafıza Değişkenleri
onceki_hata_x = 0.0
onceki_hata_y = 0.0
integral_x = 0.0
integral_y = 0.0

def sistemi_tamamen_sifirla():
    """Sistem sıfırlandığında PID integral geçmişini de temizler"""
    global kilitli, kilit_onaylandi, kilit_suresi_sayaci, roi_kutu
    global onceki_merkez_x, onceki_merkez_y, hiz_x, hiz_y, tahmin_modu, tahmin_sayaci
    global onceki_hata_x, onceki_hata_y, integral_x, integral_y
    kilitli = False
    kilit_onaylandi = False
    kilit_suresi_sayaci = 0
    roi_kutu = None
    onceki_merkez_x = None
    onceki_merkez_y = None
    hiz_x, hiz_y = 0.0, 0.0
    tahmin_modu = False
    tahmin_sayaci = 0
    onceki_hata_x = 0.0
    onceki_hata_y = 0.0
    integral_x = 0.0
    integral_y = 0.0

while True:
    basarili_mi, frame = kamera.read()
    if not basarili_mi:
        break
        
    frame = cv2.flip(frame, 1)
    
    # ROI Alan Yönetimi
    if tahmin_modu and onceki_merkez_x is not None and onceki_merkez_y is not None:
        sanal_merkez_x = onceki_merkez_x + (hiz_x * tahmin_hiz_carpani)
        sanal_merkez_y = onceki_merkez_y + (hiz_y * tahmin_hiz_carpani)
        
        rx1 = max(0, int(sanal_merkez_x - roi_genisletme))
        ry1 = max(0, int(sanal_merkez_y - roi_genisletme))
        rx2 = min(cam_genislik, int(sanal_merkez_x + roi_genisletme))
        ry2 = min(cam_yukseklik, int(sanal_merkez_y + roi_genisletme))
        
        isleme_alani = frame[ry1:ry2, rx1:rx2]
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2) 
        cv2.putText(frame, f"ESTIMATING ({tahmin_sayaci})", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    elif kilitli and roi_kutu is not None:
        rx, ry, rw, rh = roi_kutu
        rx1 = max(0, rx - roi_genisletme)
        ry1 = max(0, ry - roi_genisletme)
        rx2 = min(cam_genislik, rx + rw + roi_genisletme)
        ry2 = min(cam_yukseklik, ry + rh + roi_genisletme)
        
        isleme_alani = frame[ry1:ry2, rx1:rx2]
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1) 
    else:
        isleme_alani = frame
        rx1, ry1 = 0, 0

    if isleme_alani is None or isleme_alani.size == 0 or (isleme_alani.shape[0] < 5 or isleme_alani.shape[1] < 5):
        sistemi_tamamen_sifirla()
        continue

    hsv = cv2.cvtColor(isleme_alani, cv2.COLOR_BGR2HSV)
    
    # Genişletilmiş kırmızı maskesi
    maske1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    maske2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    tam_maske = maske1 + maske2
    tam_maske = cv2.erode(tam_maske, None, iterations=2)
    tam_maske = cv2.dilate(tam_maske, None, iterations=2)

    konturlar, _ = cv2.findContours(tam_maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nesne_bulundu = False

    if len(konturlar) > 0:
        en_buyuk_kontur = max(konturlar, key=cv2.contourArea)
        
        if cv2.contourArea(en_buyuk_kontur) > 300:
            nesne_bulundu = True
            kilitli = True
            tahmin_modu = False  
            tahmin_sayaci = 0
            
            if not kilit_onaylandi:
                kilit_suresi_sayaci += 1
                if kilit_suresi_sayaci >= gerekli_onay_frame:
                    kilit_onaylandi = True
            
            lx, ly, lw, lh = cv2.boundingRect(en_buyuk_kontur)
            x = lx + rx1; y = ly + ry1; w = lw; h = lh
            roi_kutu = [x, y, w, h]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if kilit_onaylandi:
                cv2.putText(frame, "TARGET LOCKED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"LOCKING ({int((kilit_suresi_sayaci/gerekli_onay_frame)*100)}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            merkez_x = x + w / 2
            merkez_y = y + h / 2
            
            # Filtrelenmiş Hız Vektörü Hesabı
            if onceki_merkez_x is not None and onceki_merkez_y is not None:
                anlik_hiz_x = merkez_x - onceki_merkez_x
                anlik_hiz_y = merkez_y - onceki_merkez_y
                hiz_x = (anlik_hiz_x * 0.2) + (hiz_x * 0.8)
                hiz_y = (anlik_hiz_y * 0.2) + (hiz_y * 0.8)
                if abs(hiz_x) < 1.5: hiz_x = 0.0
                if abs(hiz_y) < 1.5: hiz_y = 0.0
            
            onceki_merkez_x = merkez_x
            onceki_merkez_y = merkez_y
            
            hata_x = merkez_x - (cam_genislik / 2)
            hata_y = merkez_y - (cam_yukseklik / 2)
            
            # --- FULL PID ALGORİTMASI (TAKİP MODU) ---
            # 1. Proportional (Oransal) Terim
            p_terim_x = hata_x * kp_x
            p_terim_y = hata_y * kp_y
            
            # 2. Derivative (Türev) Terim
            turev_x = hata_x - onceki_hata_x
            turev_y = hata_y - onceki_hata_y
            d_terim_x = turev_x * kd_x
            d_terim_y = turev_y * kd_y
            
            # 3. Integral Terim & Anti-Windup Sınırlaması
            integral_x += hata_x
            integral_y += hata_y
            integral_x = max(-500, min(500, integral_x)) # İntegral patlamasını engellemek için sınır
            integral_y = max(-500, min(500, integral_y))
            i_terim_x = integral_x * ki_x
            i_terim_y = integral_y * ki_y
            
            # Servo yeni açılarının hesaplanması
            yaw += (p_terim_x + d_terim_x + i_terim_x)
            pitch -= (p_terim_y + d_terim_y + i_terim_y)
            
            onceki_hata_x = hata_x
            onceki_hata_y = hata_y

    # --- TAHMİN (ESTIMATION) LOGİC ---
    if not nesne_bulundu and kilitli and kilit_onaylandi:
        if not tahmin_modu:
            tahmin_modu = True
            tahmin_sayaci = 0
            
        tahmin_sayaci += 1
        
        onceki_merkez_x += (hiz_x * tahmin_hiz_carpani)
        onceki_merkez_y += (hiz_y * tahmin_hiz_carpani)
        
        hata_x = onceki_merkez_x - (cam_genislik / 2)
        hata_y = onceki_merkez_y - (cam_yukseklik / 2)
        
        # --- PID ALGORİTMASI (TAHMİN MODU) ---
        p_terim_x = hata_x * kp_x
        p_terim_y = hata_y * kp_y
        
        turev_x = hata_x - onceki_hata_x
        turev_y = hata_y - onceki_hata_y
        d_terim_x = turev_x * kd_x
        d_terim_y = turev_y * kd_y
        
        # Tahmin modunda yeni integral eklemiyoruz, mevcut integrali koruyoruz
        i_terim_x = integral_x * ki_x
        i_terim_y = integral_y * ki_y
        
        yaw += (p_terim_x + d_terim_x + i_terim_x)
        pitch -= (p_terim_y + d_terim_y + i_terim_y)
        
        onceki_hata_x = hata_x
        onceki_hata_y = hata_y
        
        cv2.arrowedLine(frame, (int(onceki_merkez_x - hiz_x*3), int(onceki_merkez_y - hiz_y*3)), 
                        (int(onceki_merkez_x + hiz_x * 3), int(onceki_merkez_y + hiz_y * 3)), (0, 165, 255), 2)

        if tahmin_sayaci >= maks_tahmin_frame:
            sistemi_tamamen_sifirla()

    # Sınır kontrolü ve donanımsal sıfırlamalar
    yaw = max(30, min(150, yaw))
    pitch = max(30, min(150, pitch))

    if tahmin_modu and (yaw == 30 or yaw == 150 or pitch == 30 or pitch == 150):
        sistemi_tamamen_sifirla()

    # Radar Taraması
    if not nesne_bulundu and not tahmin_modu:
        sistemi_tamamen_sifirla() 
        
        yaw += tarama_hizi_yaw * yaw_yon
        if yaw >= 150:
            yaw = 150; yaw_yon = -1; pitch += tarama_hizi_pitch * pitch_yon
        elif yaw <= 30:
            yaw = 30; yaw_yon = 1; pitch += tarama_hizi_pitch * pitch_yon
            
        if pitch >= 150:
            pitch = 150; pitch_yon = -1
        elif pitch <= 30:
            pitch = 30; pitch_yon = 1
            
        yaw = max(30, min(150, yaw))
        pitch = max(30, min(150, pitch))

    # Seri veri throttle kontrolü (~25 FPS komut gönderimi)
    su_an = time.time()
    if su_an - son_komut_zamani > 0.04:
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