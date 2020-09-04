# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
from PIL import Image

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2 import model_zoo

# Classes of object detection
subset = ['Toilet',
         'Swimming pool',
         'Bed',
         'Billiard table',
         'Sink',
         'Fountain',
         'Oven',
         'Ceiling fan',
         'Television',
         'Microwave oven',
         'Gas stove',
         'Refrigerator',
         'Kitchen & dining room table',
         'Washing machine',
         'Bathtub',
         'Stairs',
         'Fireplace',
         'Pillow',
         'Mirror',
         'Shower',
         'Couch',
         'Countertop',
         'Coffeemaker',
         'Dishwasher',
         'Sofa bed',
         'Tree house',
         'Towel',
         'Porch',
         'Wine rack',
         'Jacuzzi']

# Put target classes in alphabetical order (required for the labels being generated)
subset.sort()

# Get the trained model
MODEL_FILE = "retinanet_model_final/retinanet_model_final.pth"

@st.cache(allow_output_mutation=True)
def create_predictor( model_weights, threshold):
    """
    Loads a Detectron2 model based on model_config, threshhold and creates a default
    Detectron2 predictor.

    Returns Detectron2 default predictor and model config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.RETINANET.NUM_CLASSES = 30
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)

    return cfg, predictor



def make_inference(image, model_weights, threshold, n=5, save=False):
  """
  Makes inference on image (single image) using model_config, model_weights and threshold.

  Returns image with n instance predictions drawn on.

  Params:
  -------
  image (str) : file path to target image
  model_weights (str) : file path to model weights 
  threshold (float) : confidence threshold for model prediction, default 0.5
  n (int) : number of prediction instances to draw on, default 5
    Note: some images may not have 5 instances to draw on depending on threshold,
    n=5 means the top 5 instances above the threshold will be drawn on.
  save (bool) : if True will save image with predicted instances to file, default False
  """
  # Create predictor and model config
  cfg, predictor = create_predictor(model_weights, threshold)

  # Convert PIL image to array
  image = np.asarray(image)
  
  # Create metadata
  metadata = Metadata()
  metadata.set(thing_classes = subset)
  
  # Create visualizer instance
  visualizer = Visualizer(img_rgb=image,
                          metadata=metadata,
                          scale=0.3)
  outputs = predictor(image)
  
  # Get instance predictions from outputs
  instances = outputs["instances"]

  # Draw on predictions to image
  vis = visualizer.draw_instance_predictions(instances[:n].to("cpu"))

  return vis.get_image(), instances[:n]

def main():
    st.title("Giats Object Detection")
    st.write("## How does it work?")
    st.write("Add an image of a room and a machine learning learning model will look at it and find the objects like the example below:")
    st.image(Image.open("images/example-object-detection.png"), 
             caption="Example of model being run on a bedroom.", 
             use_column_width=True)
    st.write("## Upload your own image")
    st.write("**Note:** The model has been trained on typical household rooms and therefore will only with those kind of images.")
    uploaded_image = st.file_uploader("Choose a png or jpg image", 
                                      type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        n_boxes_to_draw = st.slider(label="Number of object to detect (boxes to draw)",
                                    min_value=1, 
                                    max_value=10, 
                                    value=5)
        
        threshold = st.slider(label="Number of confidence threshold for model prediction",
                                    min_value=0.1, 
                                    max_value=1.0, 
                                    value=0.5)

        # Make sure image is RGB
        image = image.convert("RGB")
        
        if st.button("Make a prediction"):
          with st.spinner("Please wait..."):
            custom_pred, preds = make_inference(
                image=image,
                model_weights=MODEL_FILE,
                n=n_boxes_to_draw,
                threshold=threshold
            )
            st.image(custom_pred, caption="Objects detected.", use_column_width=True)
          classes = np.array(preds.pred_classes)
          st.write("Objects detected:")
          st.write([subset[i] for i in classes])
    

if __name__ == "__main__":
    main()