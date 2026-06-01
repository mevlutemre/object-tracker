import cv2
import numpy as np
import serial
import time

# Seri haberleşme başlat (Kendi portuna göre değiştir: Windows için 'COM3', Linux için '/dev/ttyUSB0' gibi)
# Donanım bağlı değilse test edebilmen için try-except içine aldım.
try:
    ser = serial.Serial('COM6', 57600, timeout=0.1) 
    seri_acik = True
except:
    print("Seri port açılamadı! Komutlar sadece ekrana yazdırılacak.")
    seri_acik = False

def servo_sur(yaw_aci, pitch_aci):
    # İstenen xxx-yyy formatını tam 3 haneli olacak şekilde (090-120 gibi) oluştur
    komut = f"{int(yaw_aci):03d}-{int(pitch_aci):03d}\n"
    if seri_acik:
        ser.write(komut.encode())
    return komut

kamera = cv2.VideoCapture(0)

# Kameranın çözünürlüğünü al (Merkezi bulmak için gerekecek)
cam_genislik = int(kamera.get(3))
cam_yukseklik = int(kamera.get(4))

# Başlangıç açıları ve tarama yönleri
yaw = 90.0
pitch = 90.0
yaw_yon = 1     # 1: Açı artıyor, -1: Açı azalıyor
pitch_yon = 1

# Tarama hızları (Derece/Frame)
tarama_hizi_yaw = 3.0
tarama_hizi_pitch = 10.0

# Seri port gecikmesi için zamanlayıcı (Throttle)
son_komut_zamani = time.time()

while True:
    basarili_mi, frame = kamera.read()
    if not basarili_mi:
        break
        
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
            x, y, w, h = cv2.boundingRect(en_buyuk_kontur)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            merkez_x = x + w / 2
            merkez_y = y + h / 2
            
            # Nesneyi ekranın tam ortasına (hedefe) getirmek için hata miktarını bul
            hata_x = merkez_x - (cam_genislik / 2)
            hata_y = merkez_y - (cam_yukseklik / 2)
            
            # Bulduğu nesneyi takip etmesi için çok basit bir P (Oransal) kontrolcü
            # Hassasiyeti değiştirmek istersen 0.03 değerlerini kendi sistemine göre ufak ufak artırıp azaltabilirsin.
            yaw += hata_x * 0.01   
            pitch -= hata_y * 0.01 

    # Eğer ekranda kırmızı bir şey yoksa radar gibi taramaya başla
    if not nesne_bulundu:
        yaw += tarama_hizi_yaw * yaw_yon
        
        # Yaw sağ veya sol uca ulaştığında yön değiştir ve Pitch'i bir kademe indir/kaldır
        if yaw >= 150:
            yaw = 150
            yaw_yon = -1
            pitch += tarama_hizi_pitch * pitch_yon
        elif yaw <= 30:
            yaw = 30
            yaw_yon = 1
            pitch += tarama_hizi_pitch * pitch_yon
            
        # Pitch de alt veya üst uca ulaştığında yön değiştirsin
        if pitch >= 150:
            pitch = 150
            pitch_yon = -1
        elif pitch <= 30:
            pitch = 30
            pitch_yon = 1

    # Hesaplanmış olan açılar kazara 30-150 sınırının dışına çıkarsa zorla içeride tut
    yaw = max(30, min(150, yaw))
    pitch = max(30, min(150, pitch))

    # Seri porta komut gönderme ve ekrana yazdırma
    su_an = time.time()
    # Veri gönderimini saniyede ~20 komut ile sınırla (0.05 saniye gecikme)
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