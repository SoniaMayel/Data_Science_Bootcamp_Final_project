import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('/Users/sadiakhanrupa/Bootcamp Main Phase/Dr.-Greenthumb-Decoding-Nature-s-Needs/project_notebooks/models/mutil_class_classification_model.h5')

# Function to capture image from webcam
def take_photo_plant():
    # Connect to webcam
    cap = cv2.VideoCapture(0)
    # Capture a single frame
    ret, frame = cap.read()
    # Save the image
    cv2.imwrite('webcamphoto_plant.jpg', frame)
    # Releases the webcam
    cap.release()
    # Closes the frame
    cv2.destroyAllWindows()

    # Release the webcam
    cap.release()
    # Close the frame
    cv2.destroyAllWindows()

# Streamlit layout
st.title('Plant Disease Detection')

st.write("""
### Project Description
We have trained several models to predict the disease of a plant.
""")


# Button to trigger image capture
if st.button('Capture Image'):
    take_photo_plant()
    st.write('Image captured successfully!')

    # Read and preprocess the captured image
    image_cam = cv2.imread('webcamphoto_plant.jpg')
    resize = tf.image.resize(image_cam, (256, 256))
    resize_normalized = resize / 255.0  # Normalize pixel values
    resize_normalized = np.expand_dims(resize_normalized, axis=0)  # Add batch dimension

    # Class labels
    class_labels = ['Tomato___spider_mites',
 'Strawberry___healthy',
 'Grape___black_rot',
 'Potato___early_blight',
 'Tomato___leaf_curl',
 'Tomato___mosaic_virus',
 'Blueberry___healthy',
 'Sugercane___healthy',
 'Grape___leaf_blight',
 'Cherry___powdery_mildew',
 'Tomato___target_spot',
 '.DS_Store',
 'Peach___healthy',
 'Potato___late_blight',
 'Apple___rust',
 'Tomato___late_blight',
 'Tomato___leaf_mold',
 'Apple___alternaria_leaf_spot',
 'Sugercane___red_rot',
 'Potato___pests',
 'Cassava___healthy',
 'Tomato___bacterial_spot',
 'Grape___healthy',
 'Sugercane___mosaic',
 'Rice___bacterial_blight',
 'Orange___citrus_greening',
 'Tomato___early_blight',
 'Apple___scab',
 'Bell_pepper___bacterial_spot',
 'Potato___bacterial_wilt',
 'Raspberry___healthy',
 'Rice___blast',
 'Tomato___healthy',
 'Corn___northern_leaf_blight',
 'Rice___brown_spot',
 'Cassava___mosaic_disease',
 'Sugercane___rust',
 'Cassava___brown_streak_disease',
 'Cherry___healthy',
 'Rice___tungro',
 'Grape___black_measles',
 'Cassava___bacterial_blight',
 'Apple___brown_spot',
 'Corn___common_rust',
 'Cassava___green_mottle',
 'Bell_pepper___healthy',
 'Peach___bacterial_spot',
 'Apple___gray_spot',
 'Potato___virus',
 'Potato___nematode',
 'Potato___phytophthora',
 'Tomato___septoria_leaf_spot',
 'Corn___healthy',
 'Squash___powdery_mildew',
 'Corn___gray_leaf_spot',
 'Apple___black_rot',
 'Sugercane___yellow_leaf',
 'Apple___healthy',
 'Strawberry___leaf_scorch',
 'Potato___healthy',
 'Soybean___healthy']


    # Make prediction
    prediction_new = model.predict(resize_normalized)
    predicted_class_index = np.argmax(prediction_new, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    # Display prediction result
    st.write("Predicted Plant Disease:", predicted_class_label)

    # Display captured image
    st.image(image_cam, caption='Captured Image', use_column_width=True)
