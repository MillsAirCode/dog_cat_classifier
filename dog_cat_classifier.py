import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import logging
import time
from PIL import Image

# --- Setup Logging ---
logging.basicConfig(
    filename="/mnt/c/users/brad-/downloads/project_folder/dog_cat_classifier.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Image Conversion and Scaling ---
def convert_and_scale_images(input_dir, output_dir, img_size=(224, 224)):
    """Convert non-JPG/PNG images to JPG and resize all images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            logging.info(f"Skipping {filename} (already converted)")
            print(f"Skipping {filename} (already converted)")
            continue
        
        try:
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                temp_path = os.path.join(output_dir, f"temp_{output_filename}")
                img.save(temp_path, 'JPEG', quality=95)
            
            img_cv = cv2.imread(temp_path)
            if img_cv is None:
                raise ValueError("Could not load converted image")
            img_resized = cv2.resize(img_cv, img_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, img_resized)
            os.remove(temp_path)
            
            logging.info(f"Converted and resized {filename} to {output_filename} ({img_size})")
            print(f"Converted and resized {filename} to {output_filename}")
        except Exception as e:
            logging.warning(f"Failed to process {filename}: {e}")
            print(f"Warning: Failed to process {filename}: {e}")

# --- Model Definition ---
def create_model(input_shape=(224, 224, 3)):
    """Create a simple CNN model for classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Training with Augmentation ---
def train_model(base_dir, epochs=20, batch_size=32):
    """Train the model using directory-based data generators."""
    model = create_model()
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, validation_split=0.2, rescale=1./255
    )
    train_gen = datagen.flow_from_directory(
        base_dir, target_size=(224, 224), batch_size=batch_size,
        classes=['resized_dog_images', 'resized_cat_images'], class_mode='binary', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        base_dir, target_size=(224, 224), batch_size=batch_size,
        classes=['resized_dog_images', 'resized_cat_images'], class_mode='binary', subset='validation'
    )
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1)
    model.save('dog_cat_model.keras')
    logging.info("Model trained and saved as 'dog_cat_model.keras'")
    print("Model trained and saved")
    return model, history

# --- Video Classification ---
def classify_video(video_path, model, img_size=(224, 224)):
    """Classify frames in a video and determine dog or cat."""
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        print(f"Error: Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    predictions = []
    start_time = time.time()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device('/GPU:0'):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = tf.convert_to_tensor(frame, dtype=tf.float32)
                frame = tf.image.resize(frame, img_size)
                frame = frame / 255.0
                frame = tf.expand_dims(frame, axis=0)
                pred = model.predict(frame, verbose=0)[0][0]
                predictions.append(1 if pred > 0.5 else 0)
                frame_count += 1
    else:
        logging.warning("No GPU detected for video classification.")
        print("Warning: Running video classification on CPU.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)
            pred = model.predict(frame, verbose=0)[0][0]
            predictions.append(1 if pred > 0.5 else 0)
            frame_count += 1
    
    cap.release()
    
    end_time = time.time()
    inference_time = end_time - start_time
    fps = frame_count / inference_time if inference_time > 0 else 0
    
    dog_count = predictions.count(0)
    cat_count = predictions.count(1)
    total_frames = len(predictions)
    
    log_msg = (f"Analyzed {total_frames} frames in {inference_time:.2f} seconds ({fps:.2f} FPS): "
               f"Dog frames: {dog_count} ({dog_count/total_frames*100:.2f}%), "
               f"Cat frames: {cat_count} ({cat_count/total_frames*100:.2f}%)")
    logging.info(log_msg)
    print(f"Analyzed {total_frames} frames in {inference_time:.2f} seconds ({fps:.2f} FPS):")
    print(f"Dog frames: {dog_count} ({dog_count/total_frames*100:.2f}%)")
    print(f"Cat frames: {cat_count} ({cat_count/total_frames*100:.2f}%)")
    
    result = "Cat" if cat_count > dog_count else "Dog"
    logging.info(f"Dominant prediction: {result}")
    print(f"Dominant prediction: {result}")

# --- Main Execution ---
def main():
    # Define paths
    base_dir = "/mnt/c/users/brad-/downloads/project_folder"
    original_dog_dir = os.path.join(base_dir, "dog_images")
    original_cat_dir = os.path.join(base_dir, "cat_images")
    resized_dog_dir = os.path.join(base_dir, "resized_dog_images")
    resized_cat_dir = os.path.join(base_dir, "resized_cat_images")
    video_path = os.path.join(base_dir, "test_video.mp4")
    
    start_time = time.time()
    logging.info("Starting dog vs. cat classifier")
    print("Starting...")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    logging.info(f"GPUs available: {gpus}")
    print(f"GPUs available: {gpus}")
    if not gpus:
        logging.warning("No GPU detected. Running on CPU.")
        print("Warning: No GPU detected. Running on CPU.")
    
    # Convert and scale images (one-time preprocessing)
    logging.info("Converting and scaling dog images...")
    print("Converting and scaling dog images...")
    convert_and_scale_images(original_dog_dir, resized_dog_dir)
    logging.info("Converting and scaling cat images...")
    print("Converting and scaling cat images...")
    convert_and_scale_images(original_cat_dir, resized_cat_dir)
    
    # Train the model
    logging.info("Training model...")
    print("Training model...")
    model, history = train_model(base_dir)
    
    # Plot training history
    try:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("accuracy_plot.png")
        logging.info("Training plot saved as 'accuracy_plot.png'")
        print("Training plot saved as 'accuracy_plot.png'")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        print(f"Warning: Could not save plot: {e}")
    
    # Test on video
    logging.info("Classifying video...")
    print("Classifying video...")
    classify_video(video_path, model)
    
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Completed in {total_time:.2f} seconds")
    print(f"Completed in {total_time:.2f} seconds")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()