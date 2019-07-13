import ipdb
import os
import json
import subprocess
import cv2
import time

subprocess.call(['unzip','annotation.zip'])
subprocess.call(['mkdir','-p','images'])

jsonFld = './annotation/'

def save_img(fileName,page_no,loc,ImgName):
    orgimage = cv2.imread(fileName+str(page_no+1)+'.jpg')
    crop_img = orgimage[loc[1]:loc[1]+loc[3], loc[0]:loc[0]+loc[2]]
    cv2.imwrite('./images/'+ImgName,crop_img) 

cnt = 0
jsnFiles = os.listdir(jsonFld)
for item in jsnFiles:
    data = json.load(open(jsonFld+item))
    downLoad = data['url'].replace('\"','')
    dwndr = downLoad.split(' ')
    if len(item.split('_')[0])==4: # if the pdf file from the IEEE
        time.sleep(60) # waite for few seconds before download
    print ('Dowloading the file')
    subprocess.call(dwndr)
    print ('Convert the pdf file to image')
    subprocess.call(['java','-jar', '/mnt/1/pdffigureAnnotation/pdfbox-app-2.0.9.jar', 
                            'PDFToImage', '-dpi', '144', dwndr[-1]])
    regions = data['region']
    fileName = dwndr[-1].split('.pdf')[0]
    for reg in regions:
        cnt = cnt+1
        label = reg['Label']
        imgName = reg['FileName']
        pageNo = reg['PageNo']
        x = reg['boundary']['x']
        y = reg['boundary']['y']
        w = reg['boundary']['w']
        h = reg['boundary']['h']
        save_img(fileName,pageNo,(x,y,w,h),imgName)
print ('total images : ',cnt)
