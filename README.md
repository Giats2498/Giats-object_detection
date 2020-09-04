# Object Detection with Detectron2
![example of how objects might be detected in an image](https://github.com/Giats2498/Giats-object_detection2/blob/master/app/images/example-object-detection.png)
## Live version: https://giats-object-detection-288115.ew.r.appspot.com

## What's object detection?

Object detection but for common and useful household items someone looking for an Airbnb rental might want to know about. For example, does this home have a fireplace?

Original inspiration was drawn from the article [Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e).

## What's in this repo?

* The notebook with all the steps I took to train the full model. [Colab Notebook](https://colab.research.google.com/drive/1qxtzdpotxa8GvuSKOR1wqS9h9noy4MvQ?usp=sharing)
* `preprocessing.py` contains the preprocessing functions for turning [Open Images images & labels](https://storage.googleapis.com/openimages/web/index.html) into [Detectron2 style](https://detectron2.readthedocs.io/tutorials/datasets.html).
* `downloadOI.py` is a slightly modified downloader script from [LearnOpenCV](https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/) which downloads only certain classes of images from Open Images, example:

```
# Download only images from the Kitchen & dining room table class from Open Images validation set
!python3 downloadOI.py --dataset "validation" --classes "Kitchen & dining room table"
```
* `app` contains a Python script with a [Streamlit](https://www.streamlit.io) app built around the model, if you can see the live version, Streamlit is what I used to build it.
