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

    # Finde die Kanten im Bild
    edges = cv2.Canny(gauss, settings[0]/5, 1)

    return gray, gauss, edges

def run(image, result, settings=(100,100)):
    # Finde Kanten im Bild
    gray, gauss, edges = find_edges(image,settings)

    # Füge das Ergebnis dem result hinzu
    result.append({"name":f"Gray","data":gray})
    result.append({"name":f"Gauss","data":gauss})
    result.append({"name":f"Edges {settings[0]/5}","data":edges})

    return result


# Nur ausführen, wenn das Skript direkt aufgerufen wird
if __name__ == "__main__":
    
    files = get_images()
    output_folder = get_output_folder()

