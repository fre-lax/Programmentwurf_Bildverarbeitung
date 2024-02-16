import numpy as np
import cv2


def run(image, result, settings=(100,100)):

    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)
    predicted=predict_svm(svm, image)
    color_predicted=color_predict(predicted,colors)
    gray = gray_image(color_predicted)
    thresholed_image,contours = find_contours(gray,settings)
    merged_contours_img=fill_largest_rectangle(thresholed_image, contours,settings)
    final_selction,box=find_final_rectangle(merged_contours_img,image,settings)

    cropped=crop_and_rotate(image,box)

    result.append({"name":f"KI Predicted","data":predicted})
    result.append({"name":f"KI Color Predicted","data":color_predicted})
    result.append({"name":f"KI Color Predicted_gray","data":gray})
    result.append({"name":f"Contours","data":thresholed_image})
    result.append({"name":f"Merged Contours","data":merged_contours_img})
    result.append({"name":f"final selection","data":final_selction})
    result.append({"name":f"cropped","data":cropped})


def gray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def init_svm():
    # init SVM
    svm=cv2.ml.SVM_create()
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
    hintergrund_hell = [74, 74, 74]
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
    colored_image=color[(image.copy()*2).astype(int)]
    # make class 0 and 2 white
    colored_image[(image==0)]=255
    colored_image[(image==1)]=255
    return colored_image
    
def find_contours(image,settings):
    _, thresholed_image = cv2.threshold(image, 90,75, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresholed_image = np.zeros((thresholed_image.shape[0],thresholed_image.shape[1],3), dtype=np.uint8)
    cv2.drawContours(thresholed_image, contours, -1, (255,0,0), 1)
    # remove contours that are too small
    biggest = [c for c in contours if cv2.contourArea(c) > settings[0]]
    # biggest = max(contours, key=cv2.contourArea)
    cv2.drawContours(thresholed_image, biggest, -1, (0,255,0), 1)
    

    return thresholed_image,biggest

def set_training_area():
    pass

def fill_largest_rectangle(image, contours,settings):
    rect_img=image.copy()
    # find rectangles
    rects = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(box)
    # draw rectangles
    for r in rects:
        cv2.drawContours(rect_img, [r], 0, (0,0,255), 3)

    merge_contours =np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8)
    # set complete image to 50 50 50
    merge_contours[:,:,:]=50
    # get largest rectangle 
    max_rect = max(rects, key=cv2.contourArea)
    cv2.fillPoly(merge_contours, pts =[max_rect], color=(255,255,255))
    # draw all contours
    cv2.fillPoly(merge_contours, pts =contours, color=(255,255,255))
    return merge_contours

def find_final_rectangle(merged_contours_img,original_image,settings):
    final_contours_img = merged_contours_img.copy()
    # find contours in new image
    final_contours_img = cv2.cvtColor(final_contours_img, cv2.COLOR_BGR2GRAY)
    final_contours_img, contours = find_contours(final_contours_img,settings)
    

    cv2.drawContours(final_contours_img, contours, -1, (0,255,255), 1)
    #find largest contour
    biggest = max(contours, key=cv2.contourArea)
    # find rectangle in new image
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    final_box=original_image.copy()
    cv2.drawContours(final_box, [box], 0, (255,255,0), 1)

    return final_box, box
    

def crop_and_rotate(image, rect):
    box = np.int0(rect)
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


if __name__ == '__main__':
    pass