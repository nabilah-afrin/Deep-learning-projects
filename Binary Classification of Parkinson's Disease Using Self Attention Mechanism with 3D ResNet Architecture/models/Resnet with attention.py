from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, ReLU, Dense, GlobalAveragePooling3D, Add
def residual_block(x, filters, kernel_size=3, strides=1, use_attention=True):
    # Convolutional layers
    residual = x
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.2))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    if use_attention:
        x = attention_block(x)

    x = Conv3D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if strides > 1:
        residual = Conv3D(filters, kernel_size=1, strides=strides, padding='same')(residual)

    # Residual connection
    x = Add()([residual, x])
    x = ReLU()(x)
    return x

def attention_block(x):
    channels = x.shape[-1]
    attention = Dense(channels, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.2))(x)
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

    x = GlobalAveragePooling3D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model


input_shape = (128, 128, 64, 1)
num_classes = 2
model = build_3d_resnet(input_shape, num_classes, use_attention=True)

model.summary()
