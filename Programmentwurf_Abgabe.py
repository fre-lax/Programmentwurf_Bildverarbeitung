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
    rect=find_rectangle(thresholed_image, contours)

    result.append({"name":f"KI Predicted","data":predicted})
    result.append({"name":f"KI Color Predicted","data":color_predicted})
    result.append({"name":f"KI Color Predicted_gray","data":gray})
    result.append({"name":f"Contours","data":thresholed_image})
    result.append({"name":f"Rectangle","data":rect})


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

def find_rectangle(image, contours):
    # points = [pt for contour in contours for pt in contour]

    # # Ensure points is a proper sequence for np.vstack
    # points = np.array(points).reshape(-1, 2) # Reshape if necessary

    # # Now, use np.vstack to stack. Since points is already a proper numpy array, this step might be redundant
    # all_points = np.vstack(points)
    # all_points = np.array(all_points, dtype=np.float32).reshape((-1, 1, 2))
    # # add all neighboring points to the list
    # for i in range(len(points)):
    #     all_points = np.vstack((all_points, np.array([points[i-1],points[i]], dtype=np.float32).reshape((-1, 1, 2)))    )

    all_points = contours
    rect_img=image.copy()
    # find rectangles
    rects = []
    for c in all_points:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rects.append(box)
    # draw rectangles
    for r in rects:
        cv2.drawContours(rect_img, [r], 0, (0,0,255), 3)
    return rect_img
    

if __name__ == '__main__':
    pass