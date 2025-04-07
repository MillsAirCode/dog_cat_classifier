import cv2
import numpy as np
import tensorflow as tf
import time
import logging

# --- Configure GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# --- Setup Logging ---
logging.basicConfig(
    filename="/home/brad-desk/project_folder/webcam_dog_cat.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Load TensorRT Model ---
def load_trt_model(model_path):
    model = tf.saved_model.load(model_path)
    logging.info(f"TensorRT model loaded from {model_path}")
    print(f"TensorRT model loaded from {model_path}")
    return model

# --- Preprocess on GPU ---
@tf.function
def preprocess_frame(frame, img_size=(256, 256)):
    frame_tensor = tf.cast(frame, tf.float32) / 255.0
    frame_resized = tf.image.resize(frame_tensor, img_size, method='bilinear')
    return tf.expand_dims(frame_resized, axis=0)

# --- Real-Time Webcam Classification ---
def classify_webcam_realtime(model, img_size=(256, 256)):
    infer = model.signatures['serving_default']
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.time()
        
        frame_input = preprocess_frame(frame, img_size)
        t2 = time.time()
        
        pred = infer(frame_input)['output_0'][0][0].numpy()  # Adjust if needed
        label = "Dog" if pred < 0.5 else "Cat"
        confidence = 1 - pred if pred < 0.5 else pred
        t3 = time.time()
        
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        
        h, w = frame.shape[:2]
        box_size = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        x1, y1 = center_x - box_size // 2, center_y - box_size // 2
        x2, y2 = center_x + box_size // 2, center_y + box_size // 2
        color = (0, 255, 0) if label == "Dog" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label} ({confidence:.2f}) | FPS: {fps:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Dog vs Cat Classifier", frame)
        t4 = time.time()
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Capture: {(t1-t0)*1000:.1f}ms, Preprocess: {(t2-t1)*1000:.1f}ms, "
                  f"Inference: {(t3-t2)*1000:.1f}ms, Display: {(t4-t3)*1000:.1f}ms")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Processed {frame_count} frames in {(time.time() - start_time):.2f} seconds")

# --- Main ---
def main():
    model_path = "/home/brad-desk/project_folder/dog_cat_model_trt"
    model = load_trt_model(model_path)
    if model:
        classify_webcam_realtime(model)

if __name__ == "__main__":
    main()