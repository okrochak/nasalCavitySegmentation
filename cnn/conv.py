
from tensorflow.keras.layers import (
    BatchNormalization,
    Activation,
    Dropout,
)
from tensorflow.keras.layers import (
    MaxPooling2D,
    MaxPooling3D,
)
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import add
import numpy as np
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
# from skimage.transform import resize
# from sklearn.model_selection import train_test_split

# Define functions necessary for the segmentation
# Function for 2D convolutional block inside network architecture
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # First layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Second layer
    x = Conv2D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# Function for 2D network architecture
def get_net_2D(input_img, out_layer, n_filters, dropout, batchnorm=True):
    # Contracting path
    c1 = conv2d_block(
        input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling2D((2, 2))(c1)
    # p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    # p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    # p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(c5)
    # u6 = concatenate([u6, c4])
    sc6 = add([u6, c4])
    # u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(sc6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(c6)
    # u7 = concatenate([u7, c3])
    sc7 = add([u7, c3])
    # u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(sc7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(c7)
    # u8 = concatenate([u8, c2])
    sc8 = add([u8, c2])
    # u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(sc8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(c8)
    # u9 = concatenate([u9, c1], axis=3)
    sc9 = add([u9, c1])
    sc9 = Dropout(dropout)(sc9)
    c9 = conv2d_block(sc9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(out_layer, (1, 1), activation="sigmoid")(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# Function for 3D convolutional block inside network architecture
def conv3d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer
    x_1 = Conv3D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    if batchnorm:
        x_2 = BatchNormalization()(x_1)
    x_2 = Activation("relu")(x_2)

    # second layer
    x_2 = Conv3D(
        filters=n_filters,
        kernel_size=(kernel_size, kernel_size, kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(x_2)
    if batchnorm:
        x_2 = BatchNormalization()(x_2)
    x_2 = Activation("relu")(x_2)

    return x_2


# Function for 3D network architecture
def get_net_3D(input_img, n_filters, dropout, batchnorm=True):

    # contracting path
    c1 = conv3d_block(
        input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = conv3d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = conv3d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = conv3d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv3d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv3DTranspose(n_filters * 8, (3, 3, 3), strides=(2, 2, 2), padding="same")(
        c5
    )
    sc6 = add([u6, c4])
    c6 = conv3d_block(sc6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv3DTranspose(n_filters * 4, (3, 3, 3), strides=(2, 2, 2), padding="same")(
        c6
    )
    sc7 = add([u7, c3])
    c7 = conv3d_block(sc7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv3DTranspose(n_filters * 2, (3, 3, 3), strides=(2, 2, 2), padding="same")(
        c7
    )
    sc8 = add([u8, c2])
    c8 = conv3d_block(sc8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv3DTranspose(n_filters * 1, (3, 3, 3), strides=(2, 2, 2), padding="same")(
        c8
    )
    sc9 = add([u9, c1])
    sc9 = Dropout(dropout)(sc9)
    c9 = conv3d_block(sc9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv3D(1, (1, 1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    return model
