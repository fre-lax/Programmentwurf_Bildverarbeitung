import numpy as np
import cv2

def run(image, result,settings=(2,50)):
    if(len(image.shape)!=3):
            print("Nur für Farbbilder")
            return
    height, width, *_ = image.shape[:]  # image height and width

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(settings[0]/10+0.1)
    svm.setGamma(settings[0]/10.0+0.1)

    rows=np.array(image[80,59],dtype=np.float32)
    rows = np.vstack((rows,np.array(image[80,42],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[46,160],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[46,143],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[53,139],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[202,13],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[107,156],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[118,146],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[151,232],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[159,216],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[159,90],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[159,106],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[155,73],dtype=np.float32)))
    rows = np.vstack((rows,np.array(image[105,176],dtype=np.float32)))

    train = rows
    response= np.array([0,0,1,1,2,2,3,3,4,4,5,5,2,2]).astype(int) 
  
    svm.train(train, cv2.ml.ROW_SAMPLE, response)
    
    erg = svm.predict(image.reshape(height*width,3).astype(np.float32))
    erg= erg[1].reshape(height,width).astype(np.int8)
    ergColor=np.zeros((height,width,3)).astype(np.uint8)
    ergColor[erg==0]=image[80,59]
    ergColor[erg==1]=image[46,160]
    ergColor[erg==2]=image[53,139]
    ergColor[erg==3]=image[107,156]
    ergColor[erg==4]=image[159,216]
    ergColor[erg==5]=image[159,90]
    ergColor=cv2.resize(ergColor,None,None,3,3,cv2.INTER_NEAREST)

    erg = erg.reshape(height,width)/5.01

    result.append({"name":"Test1","data":erg})
    result.append({"name":"Test2","data":ergColor})

    
if __name__ == '__main__':
    image=cv2.imread("Images\Farbpunkte.jpg")
    result=[]
    run(image,result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()