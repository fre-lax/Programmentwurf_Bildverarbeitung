import numpy as np
import cv2
import tkinter as tk  #Ordner anklicken
from tkinter import filedialog #Ordner anklicken
import os #Ordner auswählen
import random
import time
import multiprocessing #Laufzeit 



def run(image, result, settings=(100,100)): #von GUI ausgeführt 
    start=time.time()
    original_image=image.copy()
    scale_factor=1                          # Bild runterskalieren, Laufzeit veränder
    if scale_factor!=1:                     
        image = cv2.medianBlur(image,1)     # Medianfilter
        image=scale_image(image,100*scale_factor)

    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)
    predicted=predict_svm(svm, image)
    color_predicted,all_classes=color_predict(predicted,colors)
    gray = gray_image(color_predicted)
    thresholed_image,contours = find_contours(gray,settings)
    merged_contours_img=fill_largest_rectangle(thresholed_image, contours,settings)
    final_selction,box=find_final_rectangle(merged_contours_img,image,settings)

    box=box/scale_factor

    cropped=crop_and_rotate(original_image,box)
    rotated=ausrichtung_korrigieren(cropped,svm)
    scaled=scale_image(rotated,50)

    result.append({"name":f"Scaled","data":image})
    result.append({"name":f"KI Predicted","data":predicted})
    result.append({"name":f"KI Color Predicted","data":all_classes})
    result.append({"name":f"KI Color Predicted, Pin class = board class = white","data":color_predicted})
    result.append({"name":f"KI Color Predicted_gray","data":gray})
    result.append({"name":f"Contours","data":thresholed_image})
    result.append({"name":f"Merged Contours","data":merged_contours_img})
    result.append({"name":f"final selection","data":final_selction})
    result.append({"name":f"cropped","data":cropped})
    result.append({"name":f"cropped and rotated","data":rotated})
    result.append({"name":f"scaled to 50% - runtime: {round((time.time()-start)*1000)} ms","data":scaled})
    return result


def gray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #brauchen Graubild für Threshold
    return gray

def init_svm():
    # init SVM
    svm=cv2.ml.SVM_create() #Objekt svm --> Modul Machine Learning, Klasse SVM_create
    svm.setKernel(cv2.ml.SVM_LINEAR) 
    svm.setType(cv2.ml.SVM_C_SVC) # Separierung

    return svm          #Rückgabe

def set_training_pixels():
    # set training pixels
    pin_1 = [179, 193, 199]  #anhand der Beispiele lernen 
    pin_2 = [144, 154, 164]
    blau_dunkel = [94, 54,  2]
    blau_hell = [187, 120,  11]
    hintergrund_dunkel = [50, 53, 51]
    hintergrund_hell = [74, 74, 74]
    hintergrund_hell = [132, 130, 129]

    colors=[pin_1,pin_2,blau_dunkel,blau_hell,hintergrund_dunkel,hintergrund_hell]
    labels=[0,0,1,1,2,2]

    train_data=np.array(colors,dtype=np.float32)    # Arrays definieren --> float --> SVM btraucht Daten
    response_data=np.array(labels,dtype=np.int32)    # Arrays definieren

    return train_data, response_data,colors,labels

def train_svm(svm, train_data, response_data):

    svm.train(train_data, cv2.ml.ROW_SAMPLE, response_data) #trainieren

    return svm

def predict_svm(svm, image):
    mat=image.reshape(image.shape[0]*image.shape[1],3)  #Liste von Pixeln untereinander und Farben, Pixel*3 ARRAY (MATRIX)
    predicted = svm.predict(mat.astype(np.float32))     # Umformen, # Hyperebene predicten,
    predicted = predicted[1].reshape(image.shape[0],image.shape[1]) # 4-Eck
    return predicted  #Klassen  (Pixel)

def color_predict(image,colors):
    color=np.array(colors).astype(np.uint8)  # formatieren
    colored_image=color[(image.copy()*2).astype(int)] # Farben 0,2,4
    # make class 0 and 2 white
    all_classes=colored_image.copy()
    colored_image[(image==0)]=255          # weiß Pin
    colored_image[(image==1)]=255           # weiß Platine
    return colored_image, all_classes
    
def find_contours(image,settings):
    _, thresholed_image = cv2.threshold(image, 90,75, cv2.THRESH_BINARY)  # Unter 90 -> schwarz, sonst weiß (andere --> 75 gesetzt)
    contours, _ = cv2.findContours(thresholed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresholed_image = np.zeros((thresholed_image.shape[0],thresholed_image.shape[1],3), dtype=np.uint8)
    cv2.drawContours(thresholed_image, contours, -1, (255,0,0), 1)
    # remove contours that are too small
    biggest = [c for c in contours if cv2.contourArea(c) > settings[0]]  #Schieberegler , Konturen>149 --> biggest
    # biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(thresholed_image, biggest, -1, (0,255,0), 1)
    

    return thresholed_image,biggest

def set_training_area():
    pass

def fill_largest_rectangle(image, contours,settings):
    rect_img=image.copy()
    # find rectangles
    rects = []     #leee Liste
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(box)
    # draw rectangles
    for r in rects:
        cv2.drawContours(rect_img, [r], 0, (0,0,255), 3)

    merge_contours =np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8)   #leeres Bild
    # get largest rectangle 
    max_rect = max(rects, key=cv2.contourArea)    # MAximale Fläche
    cv2.fillPoly(merge_contours, pts =[max_rect], color=(255,255,255))  # größtes Rechteck weiß
    # draw all contours
    cv2.fillPoly(merge_contours, pts =contours, color=(255,255,255))  #größere Konturen weiß
    return merge_contours

def find_final_rectangle(merged_contours_img,original_image,settings):  #nochmal KONTUREN finden --> 
    final_contours_img = merged_contours_img.copy()
    # find contours in new image
    final_contours_img = cv2.cvtColor(final_contours_img, cv2.COLOR_BGR2GRAY)
    final_contours_img, contours = find_contours(final_contours_img,settings)
    

    cv2.drawContours(final_contours_img, contours, -1, (0,255,255), 1)
    #find largest contour
    biggest = max(contours, key=cv2.contourArea)   #alle Punkte der größten Kontur
    # find rectangle in new image
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    box = np.int0(box)       #4 Eckpunkte
    final_box=original_image.copy()
    cv2.drawContours(final_box, [box], 0, (255,255,0), 1)

    return final_box, box
    

def crop_and_rotate(image, rect):
    box = np.int0(rect)
    # Berechne die Breite und Höhe des finalen Bildes
    width = int(np.linalg.norm(box[1]-box[2]))           #Betrag der Eckpunkte
    height = int(np.linalg.norm(box[2]-box[3]))
    
    src_pts = box.astype("float32")     #Datentyp float32 benötigt
    dst_pts = np.array([[0, height-1],          #Zielpunkte bestimmen, gleiche Reihenfolge wie bei BOX
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)           #3x3 Matrix, LGS --> 4 Punkte 
    warped = cv2.warpPerspective(image, M, (width, height))
    # check if width is bigger than height
    if width<height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def ausrichtung_korrigieren(cropped_image,svm):
    # get olnly lower third of image
    lower_third = cropped_image[int(cropped_image.shape[0]/100*90):int(cropped_image.shape[0]*92/100),:,:]
    # predict class of lower third
    predicted=predict_svm(svm, lower_third)
    # count appearences of class 1 in predicted
    counter=0
    for row in predicted:  #-> RIEHEN
        for pixel in row:
            if pixel==1:
                counter+=1
    if counter<1000:
        # rotate image 180 degree
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)
    return cropped_image

def scale_image(image,scale_percent):
    scaled=image.copy()
    width = int(scaled.shape[1] /100*scale_percent)
    height = int(scaled.shape[0] /100* scale_percent)
    dim = (width, height)
    scaled = cv2.resize(scaled, dim, interpolation = cv2.INTER_AREA)  #Durchschnittswert umliegende PIXEL
    return scaled

# Finde Bilder die verarbeitet werden sollen
def get_images():
    # Erstelle Tkinter Fenster
    window = tk.Tk()
    window.withdraw()

    # Nutzer nach Ordner fragen
    folder_selected = filedialog.askdirectory(initialdir="./pcb2/Data/Images")  #Ordner fragen, --> hinspringen
    if folder_selected == '':
        print('Kein Ordner ausgewählt')
        return '',[]
    print(f'Durchsuche Ordner nach Bildern: {folder_selected}')
    
    # Erhalte Liste aller Dateien im Ordner und seinen Unterordnern, falls Dateiendung .JPG
    files = []
    count = 0
    for r, d, f in os.walk(folder_selected): #Pfad zum Dateiname, Dateiname  selbst
        for file in f:
            if '.JPG' in file:
                files.append(os.path.join(r, file))  #Liste von Pfaden zu allen Bilddateien 
                count += 1
    print(f'Es wurden {count} Bilder gefunden.')
    return files


def verabeiten(file, output_dir):
    start=time.time()
    global svm
    global colors
    image = cv2.imread(file)
    settings=(149,255)
    predicted=predict_svm(svm, image)
    color_predicted,all_classes=color_predict(predicted,colors)
    gray = gray_image(color_predicted)
    thresholed_image,contours = find_contours(gray,settings)
    merged_contours_img=fill_largest_rectangle(thresholed_image, contours,settings)
    final_selction,box=find_final_rectangle(merged_contours_img,image,settings)
    cropped=crop_and_rotate(image,box)
    rotated=ausrichtung_korrigieren(cropped,svm)
    scaled=scale_image(rotated,50)
    save_image(scaled, output_dir, file) #Scaled--> ausgeschnittene*z,B,50%, Outpurdir: Dateipfad Ordner, wo ich speichern will,File: Ursprungsbilder, 
    print(f'Verarbeitung von {file.split("/")[-1]} dauerte {(time.time()-start)*1000} ms.')
    

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

def save_image(image, output_dir, file):
    filename = os.path.basename(file)     #kompletter Pfad
    filename = filename.split('.')[0]      #Datei ---> o.Eintrag z.B. 001, 002 anstatt z.B. 002.jpg
    if "Anomaly" in file:
        subfolder = "Anomaly"    #Unterordner
    elif "Normal" in file:
        subfolder = "Normal"     #Unterordner
    else:
        subfolder = "Unknown"

    if random.random() < 0.1:
        subfolder='test/'+subfolder
    else:
        subfolder='lernen/'+subfolder

    save_path = f'{output_dir}/{subfolder}'  #Pfad zusammensetzen

    if not os.path.exists(save_path):                #Ordnerstruktur automatisch erstellen
        os.makedirs(save_path)

    cv2.imwrite(f'{save_path}/{filename}_crop.png', image)  #Speichern, ohne jpg

def init_worker():                                                       #STart: Trainieren SVM, ansonsten für jedes einzeln (Faktor 3 der Zeit)
    global svm
    global colors
    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)

if __name__ == '__main__':
    svm=None
    colors=None

    start_tick = time.time()    #Zeit
    root = tk.Tk()              #Fenster  
    root.withdraw()              #Fenster nicht anzeigen, kann weg 
    files=get_images()           # Ordner aufrufen
    output_dir = get_output_folder()

    tasks = [(file, output_dir) for file in files]
    with multiprocessing.Pool(initializer= init_worker,processes=multiprocessing.cpu_count()) as pool:  #mehrer for-schleifen--> mehrere 
        results = pool.starmap(verabeiten, tasks)   #verarbeiten für jeden Task

    # Close the pool and wait for each task to complete
    pool.close()
    pool.join()    #warten bis alle Bilder bearbeitet sind

    print(f'Verarbeitung von {len(files)} Bildern dauerte {time.time()-start_tick} Sekunden.')
    print(f'Durchschnittliche Zeit pro Bild: {(time.time()-start_tick)/len(files)} Sekunden.')