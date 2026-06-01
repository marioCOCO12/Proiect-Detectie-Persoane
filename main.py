# **Echipa**: 21-E5
# **Studenti**: BOSNEANU COSMIN-ALEXANDRU, COCOȘILĂ MARIO-EMANUEL
# **Tema proiect**: D5-T1 | Recunoașterea unor elemente într-o imagine data
# **Surse folosite**:
# 1. https://docs.ultralytics.com/
# 2. https://github.com/opencv/opencv-python5

import cv2  # Importăm OpenCV pentru citirea și afișarea imaginilor
import os  # Importăm os pentru lucrul cu fișiere și directoare
from ultralytics import YOLO  # Importăm clasa YOLO din librăria ultralytics

def resize_fullscreen(imagine):
    h_ecran, w_ecran = 1080, 1920  # Rezoluția ecranului tău (schimbă dacă e diferit)
    h_img, w_img = imagine.shape[:2]

    scala = min(w_ecran / w_img, h_ecran / h_img)  # Calculăm scala păstrând proporțiile
    w_nou = int(w_img * scala)
    h_nou = int(h_img * scala)

    # Redimensionăm imaginea cu interpolare de calitate mai bună
    imagine_mare = cv2.resize(imagine, (w_nou, h_nou), interpolation=cv2.INTER_LANCZOS4)

    # Creăm un fundal negru de dimensiunea ecranului și centrăm imaginea
    fundal = cv2.copyMakeBorder(
        imagine_mare,
        (h_ecran - h_nou) // 2, (h_ecran - h_nou) // 2,
        (w_ecran - w_nou) // 2, (w_ecran - w_nou) // 2,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return fundal

def detecteaza_persoane():
    model = YOLO("yolov8n.pt")  # Încărcăm modelul YOLOv8 nano (se descarcă automat la prima rulare)
    director_imagini = "imagini"  # Numele folderului care conține imaginile de procesat
    fisier_log = "detectie_persoane.log"  # Numele fișierului în care salvăm rezultatele detecției

    # Verificăm dacă folderul cu imagini există; dacă nu, afișăm eroare și oprim programul
    if not os.path.exists(director_imagini):
        print(f"Eroare: Folderul '{director_imagini}' nu există!")
        return

    # Deschidem fișierul log în modul scriere (se suprascrie la fiecare rulare)
    with open(fisier_log, "w", encoding="utf-8") as log:

        # Creăm fereastra fullscreen o singură dată, înainte de bucla cu imagini
        cv2.namedWindow("Detectie", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detectie", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Parcurgem fiecare fișier din folderul cu imagini
        for nume_fisier in os.listdir(director_imagini):
            cale_image = os.path.join(director_imagini, nume_fisier)  # Construim calea completă către fișier
            imagine = cv2.imread(cale_image)  # Încărcăm imaginea de la calea specificată

            # Dacă fișierul nu este o imagine validă, îl sărim și trecem la următorul
            if imagine is None:
                continue

            # Rulăm detecția YOLO pe imagine; classes=0 = doar persoane, conf=0.4 = precizie minimă 40%
            # [0].boxes accesează direct cutiile de detecție din primul (și singurul) rezultat
            detectii = model(imagine, classes=0, conf=0.4, verbose=False)[0].boxes

            # Parcurgem fiecare persoană detectată în imaginea curentă
            for box in detectii:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extragem coordonatele colțurilor cutiei (stânga-sus și dreapta-jos)
                precizie = float(box.conf[0])  # Extragem scorul de încredere al detecției (între 0 și 1)
                coordonate = f"X:{x1}, Y:{y1}, Latime:{x2-x1}, Inaltime:{y2-y1}"  # Formatăm coordonatele pentru afișare

                # Afișăm în consolă informațiile despre persoana detectată
                print(f"Imagine: {nume_fisier} | Coordonate: [{coordonate}] | Precizie: {precizie:.2f}")

                # Scriem aceleași informații în fișierul log
                log.write(f"Imagine: {nume_fisier} | Coordonate: [{coordonate}] | Precizie: {precizie:.2f}\n")

                # Desenăm un dreptunghi verde în jurul persoanei detectate în imagine
                cv2.rectangle(imagine, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Afișăm în consolă numărul total de persoane detectate în imaginea curentă
            print(f"  → {len(detectii)} persoană/persoane detectată/detectate în '{nume_fisier}'")

            # Afișăm imaginea mărită și centrată pe ecran timp de 1.5 secunde
            cv2.imshow("Detectie", resize_fullscreen(imagine))
            cv2.waitKey(1500)  # Așteptăm 1500 milisecunde (1.5 secunde) înainte de a trece la următoarea imagine

    cv2.destroyAllWindows()  # Închidem toate ferestrele OpenCV deschise la finalul procesării

# Punctul de intrare în program — rulăm funcția principală doar dacă scriptul e executat direct
if __name__ == "__main__":
    detecteaza_persoane()