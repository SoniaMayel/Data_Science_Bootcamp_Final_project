# Dr. Greenthumb: Decoding Nature's Needs

🪴Early detection/prediction of tomato diseases using image-recognition models (Multi-Class Image Detection - CNN). The model predicts what the plant is suffering from. 

🤖The chatbot has an interface which allows users to ask questions, upload images, and utilize a webcam for plant diagnostics. By positioning a plant in front of the webcam and capturing an image, the system can analyze the plant for issues. 
Uploading an image is also possible. Users can select an image to upload, and if the incorrect image is chosen, they can re-upload the correct one. Upon successful detection, the chatbot offers a solution. 

## Data Set Overview
The first dataset was taken from Kaggle 
[New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data)

This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.


- I worked on this group project together with:

- [Sadia Khan Rupa](https://www.linkedin.com/in/sadia-khan-rupa/)
- [Marvin Lipps](https://www.linkedin.com/in/marvinlipps/)

 ### Use Libraries
 * Multi-Class-Image-CNN: 
 - Tensorflow - Keras
 - OpenCV
 - Numpy
 - Matplotlib
 - Seaborn
 * RAG-LLM:
 - LlamaCPP 
 - LangChain
 - RegEx
 - Streamlit

## enable GPU support in tensorflow on MACOS
Read this linkedIn Artcle
[GPU_Support](https://www.linkedin.com/pulse/how-enable-gpu-support-tensorflow-pytorch-macos-michael-hannecke-ocoye/)

## Image Classification using CNN
Reda this article
[Image classification](https://datagen.tech/guides/image-classification/image-classification-using-cnn/)

# dep learning article
[How to Learn Deep Learning from Scratch?](https://www.projectpro.io/article/learn-deep-learning/725)

[What is Deep Learning? A Tutorial for Beginners](https://www.datacamp.com/tutorial/tutorial-deep-learning-tutorial)

[Convolutional Neural Networks (CNN) with TensorFlow Tutorial](https://www.datacamp.com/tutorial/cnn-tensorflow-python)

# Youtube Tutorial
[Build a Deep CNN Image Classifier with ANY Images](https://www.youtube.com/watch?v=jztwpsIzEGc)

[Github Project for this tutorial](https://github.com/nicknochnack/ImageClassification)

### Presentation
Together with the material of my group colleagues and after rehearsal and feedback from our trainers, we condensed our findings into a five-minute presentation ([slides](https://docs.google.com/presentation/d/17wLVmW2HVz_QQxeN78b26KHGa78dyxvCeL45_Nu1le8/edit#slide=id.SLIDES_API1943556550_0)).