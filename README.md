# DocFigure

A dataset for scientific document figure classiication

## How to get the dataset

We proved the scientific document images from the article published in CVPR, ECCV and ICCV.
We don't have any copy write on this figure images.
We provide you a python script for dowloading the pdf files from [IEEE](https://ieeexplore.ieee.org/) and [CVF](http://openaccess.thecvf.com/menu.py).
_Please make sure that you have acces to these websites._

Convert the all pdf file to image file.
Download [pdfbox](http://mirrors.estointernet.in/apache/pdfbox/2.0.14/pdfbox-app-2.0.14.jar)

```javascript
git clone https://github.com/jobinkv/DocFigure.git
cd DocFigure
wget http://mirrors.estointernet.in/apache/pdfbox/2.0.14/pdfbox-app-2.0.14.jar
python readAnotation.py

```
It will create a folder sub images in a folder `images`


## Trained Models

Trained model [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/jobin_kv_research_iiit_ac_in/EYe6eejq2FhLjv8qNVoWxgwBK9aNs-aJgqem1ty6lb9-Zg?e=vdjLmP)

To test the trained model run

```javascript
python testTrainedModel.py --trainedFigClassModel '/downloded/path/to/epoch_9_loss_0.04706_testAcc_0.96867_X_resnext101_docSeg.pth' --inputImage '/path/of/inputimage/for/testing'
```
