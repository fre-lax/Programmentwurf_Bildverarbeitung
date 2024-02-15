import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os

# Definition von Funktionen

# Finde Bilder die verarbeitet werden sollen
def get_images():
    # Erstelle Tkinter Fenster
    window = tk.Tk()
    window.withdraw()

    # Nutzer nach Ordner fragen
    folder_selected = filedialog.askdirectory(initialdir="./pcb2/Data/Images")
    if folder_selected == '':
        print('Kein Ordner ausgewählt')
        return '',[]
    print(f'Durchsuche Ordner nach Bildern: {folder_selected}')
    
    # Erhalte Liste aller Dateien im Ordner und seinen Unterordnern, falls Dateiendung .JPG
    files = []
    count = 0
    for r, d, f in os.walk(folder_selected):
        for file in f:
            if '.JPG' in file:
                files.append(os.path.join(r, file))
                count += 1
    print(f'Es wurden {count} Bilder gefunden.')
    return files

# Zielordner für Bilder
def get_output_folder():
    # Erstelle Tkinter Fenster
    window = tk.Tk()
    window.withdraw()

    # Nutzer nach Ordner fragen
    folder_selected = filedialog.askdirectory(initialdir="./cropped_pcb_images/")
    if folder_selected == '':
        print('Kein Ordner ausgewählt')
        return ''
    print(f'Zielordner für Bilder: {folder_selected}')

    return folder_selected

def find_edges(image,settings):
    # Erstelle ein leeres Bild
    edges = np.zeros(image.shape, dtype=np.uint8)

    # Konvertiere das Bild in Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # weichzeichnen
    gauss = cv2.GaussianBlur(gray, (19,19),0)

    # Finde die Kanten im Bild und mache sie dicker
    edges = cv2.Canny(gauss, settings[0]/5, 1)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)


    return gray, gauss, edges

def blue_mask(image):
    height, width, _ = image.shape[:]
    image2=np.zeros((height,width,3),np.uint8)

    b, g, r = cv2.split(image)
    mask = (b >120) & (r>10)
    image2[mask] = (255,255,255) 


    return image2

def run(image, result, settings=(100,100)):
    # Finde Kanten im Bild
    blue_mask_image = blue_mask(image)
    gray, gauss, edges = find_edges(blue_mask_image,settings)
    contours,ccc = find_contours(edges)
    # Finde die kleinsten Rechtecke um die Konturen
    rect_img, rects = find_smallest_rect(contours, ccc)

    # Zeichne Rechteck ins bild
    mit_rechteck = image.copy()
    for r in rects:
        cv2.drawContours(mit_rechteck, [r], 0, (10,200,40), 3)

    # Schneide das Bild anhand des Rechtecks
    final_crop = crop_and_rotate(image, rects[0])

    # Füge das Ergebnis dem result hinzu
    result.append({"name":f"Blue Mask","data":blue_mask_image})
    result.append({"name":f"Gray","data":gray})
    result.append({"name":f"Gauss","data":gauss})
    result.append({"name":f"Edges {settings[0]/5}","data":edges})
    result.append({"name":f"Contours","data":contours})
    result.append({"name":f"Rectangle","data":rect_img})
    result.append({"name":f"Final","data":mit_rechteck})
    result.append({"name":f"Final Crop","data":final_crop})
    

    return result

def crop_and_rotate(image, rect):
    box = np.int0(rect)
    # Berechne die Breite und Höhe des Rechtecks
    width = int(np.linalg.norm(box[1]-box[2]))
    height = int(np.linalg.norm(box[2]-box[3]))
    
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (width, height))
    # check if width is bigger than height
    if width<height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def find_contours(image):
    # Finde die Konturen im Bild
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Verwerfe Konturen die zu klein sind
    contours = [c for c in contours if cv2.contourArea(c) > 3000]


    # Zeichne die Konturen in img
    img = image.copy()
    # wandle in farbiges Bild um
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (10,30,200), 3)

    return img, contours

def find_smallest_rect(image, contours):
    # Vereinige alle Konturen zu einem Rechteck
    rects = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(box)
    # Zeichne die Rechtecke in img
    img = image.copy()
    for r in rects:
        cv2.drawContours(img, [r], 0, (10,200,40), 3)
    
    # Finden des größten Rechtecks
    biggest = max(rects, key=cv2.contourArea)

    
    return img, [biggest]

# Nur ausführen, wenn das Skript direkt aufgerufen wird
if __name__ == "__main__":
    
    files = get_images()
    output_folder = get_output_folder()

    for file in files:
        # Lade das Bild
        image = cv2.imread(file)

        blue_mask_image = blue_mask(image)

        gray, gauss, edges = find_edges(blue_mask_image,[20])
        contours,ccc = find_contours(edges)
        # Finde die kleinsten Rechtecke um die Konturen
        rect_img, rects = find_smallest_rect(contours, ccc)

        # Zeichne Rechteck ins bild
        mit_rechteck = image.copy()
        for r in rects:
            cv2.drawContours(mit_rechteck, [r], 0, (10,200,40), 3)

        # Schneide das Bild anhand des Rechtecks
        final_crop = crop_and_rotate(image, rects[0])

        # Speichere das Bild
        filename = os.path.basename(file)
        filename = filename.split('.')[0]
        if "Anomaly" in file:
            subfolder = "Anomaly"
        elif "Normal" in file:
            subfolder = "Normal"
        else:
            subfolder = "Unknown"
        this_path = f'{output_folder}/{subfolder}'
        if not os.path.exists(this_path):
            os.makedirs(this_path)
        cv2.imwrite(f'{this_path}/{filename}_crop.jpg', final_crop)
