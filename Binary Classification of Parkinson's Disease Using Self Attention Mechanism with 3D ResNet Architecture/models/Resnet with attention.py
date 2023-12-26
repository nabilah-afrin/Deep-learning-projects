import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, ReLU, Add, GlobalAveragePooling3D, Dense

def residual_block(x, filters, kernel_size=3, strides=1, use_attention=True):
    # Convolutional layers
    residual = x
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Attention mechanism (if enabled)
    if use_attention:
        x = attention_block(x)

    x = Conv3D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)  # Adjust strides here
    x = BatchNormalization()(x)

    # Adjust dimensions of the residual tensor if needed
    if strides > 1:
        residual = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(residual)

    # Residual connection
    x = Add()([residual, x])
    x = ReLU()(x)
    return x

def attention_block(x):
    # Attention mechanism (you can customize this based on your needs)
    channels = x.shape[-1]
    attention = Dense(channels, activation='softmax')(x)
    x = tf.multiply(x, attention)
    return x

def build_3d_resnet(input_shape, num_classes, use_attention=True):
    input_tensor = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv3D(64, kernel_size=7, strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for _ in range(3):
        x = residual_block(x, filters=64, use_attention=use_attention)

    x = residual_block(x, filters=128, strides=2, use_attention=use_attention)
    for _ in range(3):
        x = residual_block(x, filters=128, use_attention=use_attention)

    x = residual_block(x, filters=256, strides=2, use_attention=use_attention)
    for _ in range(3):
        x = residual_block(x, filters=256, use_attention=use_attention)

    # Global Average Pooling and Dense layer for classification
    x = GlobalAveragePooling3D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model

# Example usage
input_shape = (128, 128, 64, 1)  # Adjust dimensions based on your 3D image size
num_classes = 2   # Binary classification
model = build_3d_resnet(input_shape, num_classes, use_attention=True)

# Print the model summary
model.summary()
