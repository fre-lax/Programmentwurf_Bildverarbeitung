import numpy as np
import cv2

def run(image, result,settings=(2,50)):
    trainer=cv2.imread("graphics/train_cashew.JPG")
    if(len(image.shape)!=3):
            print("Nur für Farbbilder")
            return
    img_reco=image.copy()

    height, width, *_ = img_reco.shape[:]  # image height and width

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(settings[0]/10+0.1)
    svm.setGamma(settings[0]/10.0+0.1)

    # define to corners of areas for training
    # no cashew:
    x1,y1=776,951
    x2,y2=x1+100,y1+100
    # cashew:
    x3,y3=610,820
    x4,y4=x3+100,y3+100

    # draw rectangles around areas
    # cv2.rectangle(img_reco,(y1,x1),(y2,x2),(0,0,255),2)
    # cv2.rectangle(img_reco,(y3,x3),(y4,x4),(0,0,255),2)

    # create training data
    rows=np.array(trainer[x1,y1],dtype=np.float32)
    response=[0]
    # no cashew
    for x in range(x1,x2):
        for y in range(y1,y2):
            rows = np.vstack((rows,np.array(trainer[x,y],dtype=np.float32)))
            response.append(0)

    # cashew
    for x in range(x3,x4):
        for y in range(y3,y4):
            rows = np.vstack((rows,np.array(trainer[x,y],dtype=np.float32)))
            response.append(1)
    
          
    response= np.array(response).astype(int)
    train = rows
  
    svm.train(train, cv2.ml.ROW_SAMPLE, response)
    
    erg = svm.predict(img_reco.reshape(height*width,3).astype(np.float32))
    erg= erg[1].reshape(height,width).astype(np.int8)
    # ergColor=np.zeros((height,width,3)).astype(np.uint8)
    # ergColor[erg==0]=image[80,59]
    # ergColor[erg==1]=image[46,160]
    # ergColor=cv2.resize(ergColor,None,None,3,3,cv2.INTER_NEAREST)

    erg = erg.reshape(height,width)/5.01




    # Assuming `erg` is your grayscale image
    _, thresh = cv2.threshold(erg, 0,30, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)  # Convert image to 8-bit
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
    conList=[]
    rectList=[]
    for i,ele in enumerate(contours):
        if cv2.contourArea(ele)>100000:
            # Smooth contours
            epsilon = 0.0019  * cv2.arcLength(ele, True)
            ele = cv2.approxPolyDP(ele, epsilon, True)
 
            conList.append(ele)
            rectList.append(cv2.minAreaRect(ele))

    # Draw the contours
    cv2.drawContours(img_reco, conList, -1, (0, 255, 0), 1)

    # Draw the rectangles
    for rect in rectList:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_reco, [box], 0, (0, 0, 255), 2)
    
    # get corners of rectangles
    for rect in rectList:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        for i in range(4):
            cv2.circle(img_reco, (box[i,0],box[i,1]), 5, (0,255,0), -1)
    
    # warp perspective of rectangles
    for rect in rectList:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
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

    
    result.append({"name":"only ca$hew","data":warped})
    result.append({"name":"Geometry","data":img_reco})
    result.append({"name":"Ca$h Blob","data":erg})
    # result.append({"name":"Ca$h2","data":ergColor})


def get_images():
    # ask user for folder
    import tkinter as tk
    from tkinter import filedialog
    import os
    root = tk.Tk()
    root.withdraw()
    #ask user for folder
    folder_selected = filedialog.askdirectory(initialdir="/Users/fred/Library/CloudStorage/GoogleDrive-freddy.stepp@gmail.com/My Drive/Sciebo/01_Dual-Sick-Elektrotechnik Automation/DHBW/06_WS-24/03_Digitale Bildverarbeitung/00_GitHub/03_Digitale-Bildverarbeitung/cashew/cashew/Data/Images/Normal")
    # get all files in folder
    files = os.listdir(folder_selected)
    return folder_selected,files

def train_model():
    trainer=cv2.imread("graphics/train_cashew.JPG")
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2/10+0.1)
    svm.setGamma(2/10.0+0.1)

    # define to corners of areas for training
    # no cashew:
    x1,y1=776,951
    x2,y2=x1+100,y1+100
    # cashew:
    x3,y3=610,820
    x4,y4=x3+100,y3+100

    # draw rectangles around areas
    # cv2.rectangle(img_reco,(y1,x1),(y2,x2),(0,0,255),2)
    # cv2.rectangle(img_reco,(y3,x3),(y4,x4),(0,0,255),2)

    # create training data
    rows=np.array(trainer[x1,y1],dtype=np.float32)
    response=[0]
    # no cashew
    for x in range(x1,x2):
        for y in range(y1,y2):
            rows = np.vstack((rows,np.array(trainer[x,y],dtype=np.float32)))
            response.append(0)

    # cashew
    for x in range(x3,x4):
        for y in range(y3,y4):
            rows = np.vstack((rows,np.array(trainer[x,y],dtype=np.float32)))
            response.append(1)
    
          
    response= np.array(response).astype(int)
    train = rows
  
    svm.train(train, cv2.ml.ROW_SAMPLE, response)
    return svm

def extract_cashew(svm, image):
    height, width, *_ = image.shape[:]  # image height and width
    erg = svm.predict(image.reshape(height*width,3).astype(np.float32))
    erg= erg[1].reshape(height,width).astype(np.int8)

    erg = erg.reshape(height,width)/5.01

    # Assuming `erg` is your grayscale image
    _, thresh = cv2.threshold(erg, 0,30, cv2.THRESH_BINARY)
    thresh = cv2.convertScaleAbs(thresh)  # Convert image to 8-bit
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    
    conList=[]
    rectList=[]
    for i,ele in enumerate(contours):
        if cv2.contourArea(ele)>100000:
            # Smooth contours
            epsilon = 0.0019  * cv2.arcLength(ele, True)
            ele = cv2.approxPolyDP(ele, epsilon, True)
 
            conList.append(ele)
            rectList.append(cv2.minAreaRect(ele))

    # Draw the contours
    # cv2.drawContours(image, conList, -1, (0, 255, 0), 1)

    # Draw the rectangles
    # for rect in rectList:
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    
    # get corners of rectangles
    for rect in rectList:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # for i in range(4):
        #     cv2.circle(image, (box[i,0],box[i,1]), 5, (0,255,0), -1)
    
    # warp perspective of rectangles
    for rect in rectList:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
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

def save_folder():
    # ask user for folder to save the data
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    #ask user for folder
    folder_selected = filedialog.askdirectory(initialdir="/Users/fred/Library/CloudStorage/GoogleDrive-freddy.stepp@gmail.com/My Drive/Sciebo/01_Dual-Sick-Elektrotechnik Automation/DHBW/06_WS-24/03_Digitale Bildverarbeitung/00_GitHub/03_Digitale-Bildverarbeitung/cashew/")
    return folder_selected

def save_image(image, name, folder):
    #create subfolder if not exists
    import os
    if not os.path.exists(folder+"/train"):
        os.makedirs(folder+"/train")
    if not os.path.exists(folder+"/test"):
        os.makedirs(folder+"/test")
    import random
    if random.random()<0.9:
        cv2.imwrite(folder+"/train/"+name+".jpg",image)
    else:
        cv2.imwrite(folder+"/test/"+name+".jpg",image)

if __name__ == '__main__':
    svm=train_model()
    folder,files=get_images()
    if folder=="":
        print("No source folder selected")
        exit()
    result=[]
    save_folder=save_folder()
    if save_folder=="":
        print("No target folder selected")
        exit()
    for i,f in enumerate(files):
        if ".JPG" not in f:
            continue
        print(f"Analysing {f}")
        image=cv2.imread(folder+"/"+f)
        cropped=extract_cashew(svm,image)
        # resize image to 20% of original size
        cropped = cv2.resize(cropped, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        save_image(cropped,f,save_folder) 
        if i<10:
            x=input()
            if x=="x":
                break

        # if user presses x, stop the loop
        #if cv2.waitKey(0) & 0xFF == ord('x'):
        #    break
        # any other key, close the window
        #cv2.destroyAllWindows()