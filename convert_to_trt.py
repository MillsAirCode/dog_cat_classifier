import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

model_path = "/home/brad-desk/project_folder/dog_cat_model_4090.keras"
saved_model_path = "/home/brad-desk/project_folder/dog_cat_model_saved"
trt_model_path = "/home/brad-desk/project_folder/dog_cat_model_trt"

# Load Keras model
model = tf.keras.models.load_model(model_path)

# Define a tf.function with explicit input signature
@tf.function(input_signature=[tf.TensorSpec([None, 256, 256, 3], tf.float32, name='input_1')])
def serving_fn(inputs):
    return {'output_0': model(inputs)}

# Save as SavedModel with the signature
tf.saved_model.save(
    model,
    saved_model_path,
    signatures={'serving_default': serving_fn}
)

# Convert to TensorRT
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_path,
    precision_mode=trt.TrtPrecisionMode.FP16
)
converter.convert()
converter.save(trt_model_path)

print(f"TensorRT model saved to {trt_model_path}")