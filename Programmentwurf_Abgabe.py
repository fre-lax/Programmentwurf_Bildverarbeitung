import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os
import random
import time
import multiprocessing



def run(image, result, settings=(100,100)):
    start=time.time()
    original_image=image.copy()
    scale_factor=1
    if scale_factor!=1:
        image = cv2.medianBlur(image,1)
        image=scale_image(image,100*scale_factor)

    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)
    predicted=predict_svm(svm, image)
    black_white_image,all_classes=color_predict(predicted,colors)
    gray = gray_image(black_white_image)
    contour_image,contours = find_contours(gray,settings,scale_factor)
    merged_contours_img=fill_largest_rectangle(contour_image, contours,settings)
    final_selction,box=find_final_rectangle(merged_contours_img,image,settings,scale_factor)

    box=box/scale_factor

    ausgerichtet=auschneiden_und_ausrichten(original_image,box)
    rotated=rotation_korrigieren(ausgerichtet,svm)
    scaled=scale_image(rotated,50)

    result.append({"name":f"Scaled","data":image})
    result.append({"name":f"KI-Vorhersage (gefärbt)","data":all_classes})
    result.append({"name":f"Schwarz-Weiß","data":black_white_image})
    result.append({"name":f"Alle Konturen","data":contour_image})
    result.append({"name":f"Verbundene Konturen","data":merged_contours_img})
    result.append({"name":f"Finale Auswahl","data":final_selction})
    result.append({"name":f"Ausgerichtet","data":ausgerichtet})
    result.append({"name":f"cropped and rotated","data":rotated})
    result.append({"name":f"scaled to 50% - runtime: {round((time.time()-start)*1000)} ms","data":scaled})
    return result


def gray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def init_svm():
    # init SVM
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)

    return svm

def set_training_pixels():
    # set training pixels
    pin_1 = [179, 193, 199]
    pin_2 = [144, 154, 164]
    blau_dunkel = [94, 54,  2]
    blau_hell = [187, 120,  11]
    hintergrund_dunkel = [50, 53, 51]
    hintergrund_hell = [132, 130, 129]

    colors=[pin_1,pin_2,blau_dunkel,blau_hell,hintergrund_dunkel,hintergrund_hell]
    labels=[0,0,1,1,2,2]

    train_data=np.array(colors,dtype=np.float32)
    response_data=np.array(labels,dtype=np.int32)

    return train_data, response_data,colors,labels

def train_svm(svm, train_data, response_data):

    svm.train(train_data, cv2.ml.ROW_SAMPLE, response_data)

    return svm

def predict_svm(svm, image):
    mat=image.reshape(image.shape[0]*image.shape[1],3)
    predicted = svm.predict(mat.astype(np.float32))
    predicted = predicted[1].reshape(image.shape[0],image.shape[1])
    return predicted

def color_predict(image,colors):
    color=np.array(colors).astype(np.uint8)
    black_white_image=color[(image.copy()*2).astype(int)]
    # make class 0 and 2 white
    all_classes=black_white_image.copy()
    black_white_image[(image==0)]=255
    black_white_image[(image==1)]=255
    black_white_image[(image==2)]=0
    return black_white_image, all_classes
    
def find_contours(black_white_image,settings,scale_factor):
    contours, _ = cv2.findContours(black_white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros((black_white_image.shape[0],black_white_image.shape[1],3), dtype=np.uint8)
    cv2.drawContours(contour_image, contours, -1, (255,0,0), 1)
    # remove contours that are too small
    biggest = [c for c in contours if cv2.contourArea(c) > 149*scale_factor**2]
    # biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(contour_image, biggest, -1, (0,255,0), 1)

    return contour_image,biggest

def set_training_area():
    pass

def fill_largest_rectangle(image, contours,settings):
    # find rectangles
    rects = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(box)

    merge_contours =np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8)
    # get largest rectangle 
    max_rect = max(rects, key=cv2.contourArea)
    cv2.fillPoly(merge_contours, pts =[max_rect], color=(255,255,255))
    # draw all contours
    cv2.fillPoly(merge_contours, pts =contours, color=(255,255,255))
    return merge_contours

def find_final_rectangle(merged_contours_img,original_image,settings,scale_factor):
    # find contours in new image
    merged_contours_img = cv2.cvtColor(merged_contours_img, cv2.COLOR_BGR2GRAY)
    merged_contours_img, contours = find_contours(merged_contours_img,settings,scale_factor)
    

    #find largest contour
    biggest = max(contours, key=cv2.contourArea)
    # find rectangle in new image
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    final_box_image=original_image.copy()
    cv2.drawContours(final_box_image, [biggest], -1, (0,255,255), 2)
    cv2.drawContours(final_box_image, [box], 0, (255,255,0), 2)

    return final_box_image, box
    

def auschneiden_und_ausrichten(image, box):
    # Berechne die Breite und Höhe des finalen Bildes
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

def rotation_korrigieren(cropped_image,svm):
    # get olnly lower third of image
    zwei_prozent_streifen = cropped_image[int(cropped_image.shape[0]/100*90):int(cropped_image.shape[0]*92/100),:,:]
    # predict class of lower third
    predicted=predict_svm(svm, zwei_prozent_streifen)
    # count appearences of class 1 in predicted
    counter=0
    for row in predicted:
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
    scaled = cv2.resize(scaled, (width, height), interpolation = cv2.INTER_AREA)
    return scaled

# Finde Bilder die verarbeitet werden sollen
def get_images():

    # Nutzer nach Ordner fragen
    folder_selected = filedialog.askdirectory(initialdir="./pcb2/Data/Images")
    if folder_selected == '':
        print('Kein Ordner ausgewählt')
        return None
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


def verabeiten(file, output_dir):
    start=time.time()
    global svm
    global colors
    
    start=time.time()
    settings=(149,255)
    image = cv2.imread(file)
    original_image=image.copy()

    scale_factor=1
    if scale_factor!=1:
        image = cv2.medianBlur(image,1)
        image=scale_image(image,100*scale_factor)

    predicted=predict_svm(svm, image)
    black_white_image,all_classes=color_predict(predicted,colors)
    gray = gray_image(black_white_image)
    contour_image,contours = find_contours(gray,settings,scale_factor)
    merged_contours_img=fill_largest_rectangle(contour_image, contours,settings)
    final_selction,box=find_final_rectangle(merged_contours_img,image,settings,scale_factor)

    box=box/scale_factor

    ausgerichtet=auschneiden_und_ausrichten(original_image,box)
    rotated=rotation_korrigieren(ausgerichtet,svm)
    scaled=scale_image(rotated,50)
    save_image(scaled, output_dir, file)
    print(f'Verarbeitung von {file.split("/")[-1]} dauerte {(time.time()-start)*1000} ms.')

    

# Zielordner für Bilder
def get_output_folder():
    # Nutzer nach Ordner fragen
    folder_selected = filedialog.askdirectory(initialdir="./cropped_pcb_images/")
    if folder_selected == '':
        print('Kein Ordner ausgewählt')
        return ''
    print(f'Zielordner für Bilder: {folder_selected}')

    return folder_selected

def save_image(image, output_dir, file):
    filename = os.path.basename(file)
    filename = filename.split('.')[0]
    if "Anomaly" in file:
        subfolder = "Anomaly"
    elif "Normal" in file:
        subfolder = "Normal"
    else:
        subfolder = "Unknown"

    if random.random() < 0.1:
        subfolder='test/'+subfolder
    else:
        subfolder='lernen/'+subfolder

    save_path = f'{output_dir}/{subfolder}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cv2.imwrite(f'{save_path}/{filename}_crop.png', image)

def init_worker():
    global svm
    global colors
    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)

if __name__ == '__main__':
    svm=None
    colors=None

    start_tick = time.time()

    files=get_images()
    output_dir = get_output_folder()

    tasks = [(file, output_dir) for file in files]
    with multiprocessing.Pool(initializer= init_worker,processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(verabeiten, tasks)

    # Close the pool and wait for each task to complete
    pool.close()
    pool.join()

    print(f'Verarbeitung von {len(files)} Bildern dauerte {time.time()-start_tick} Sekunden.')
    print(f'Durchschnittliche Zeit pro Bild: {(time.time()-start_tick)/len(files)} Sekunden.')