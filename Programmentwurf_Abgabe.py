import numpy as np
import cv2


def run(image, result, settings=(100,100)):

    svm=init_svm()
    train_data, response_data,colors,labels = set_training_pixels()
    svm=train_svm(svm, train_data, response_data)
    predicted=predict_svm(svm, image)
    color_predicted=color_predict(predicted,colors)

    result.append({"name":f"KI Predicted","data":predicted})
    result.append({"name":f"KI Color Predicted","data":color_predicted})


    

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
    return colored_image
    



def set_training_area():
    pass

if __name__ == '__main__':
    pass