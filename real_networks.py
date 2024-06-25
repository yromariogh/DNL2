from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate

def C0_trained(inputs):
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)


    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
    up5 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)


    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    up6 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    # up7 = Conv2DTranspose(L, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(1, 1, name='mapping')(conv7)
    return final

def C0(pretrained_weights=None, input_size=(256, 256, 31)):
    inputs = Input(input_size)
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)


    up5 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv4))
    up5 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)


    up6 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    up6 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    # up7 = Conv2DTranspose(L, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv7)

    final = Conv2D(1, 1)(conv7)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def C1_trained(inputs):
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)



    up5 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    # up5 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up5)
    merge5 = concatenate([conv4, up5], axis=3) #conv3?
    conv6 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv6 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)



    up6 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up6 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv3, up6], axis=3)
    conv7 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv7 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up7 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up7 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv2, up7], axis=3)
    conv8 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv8)

    up8 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv1, up8], axis=3)
    conv9 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    final = Conv2D(1, 1, name='mapping')(conv9)

    return final

def C1(pretrained_weights=None, input_size=(256, 256, 31)):
    inputs = Input(input_size)
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)



    up5 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    # up5 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up5)
    merge5 = concatenate([conv4, up5], axis=3) #conv3?
    conv6 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv6 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)



    up6 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up6 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv3, up6], axis=3)
    conv7 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv7 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up7 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up7 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv2, up7], axis=3)
    conv8 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv8)

    up8 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv1, up8], axis=3)
    conv9 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    final = Conv2D(1, 1, name='mapping')(conv9)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def C2_trained(inputs): ## Corregir pq se añaden 3 capas de convolucion mas y el L esta mal y los concatenate
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)



    up6 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    # up6 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv5, up6], axis=3) #conv3?
    conv7 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv7 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)



    up7 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    # up7 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv4, up7], axis=3)
    conv8 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up8 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up8 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv3, up8], axis=3)
    conv9 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv9)

    up9 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    up9 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv2, up9], axis=3)
    conv10 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    up10 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    # up10 = Conv2DTranspose(L, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv1, up10], axis=3)
    conv11 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    final = Conv2D(1, 1, name='mapping')(conv11)

    return final

def C2(pretrained_weights=None, input_size=(256, 256, 31)):
    inputs = Input(input_size)
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)



    up6 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    # up6 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up6)
    merge6 = concatenate([conv5, up6], axis=3) #conv3?
    conv7 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv7 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)



    up7 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    # up7 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv4, up7], axis=3)
    conv8 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up8 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up8 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv3, up8], axis=3)
    conv9 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv9)

    up9 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    up9 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv2, up9], axis=3)
    conv10 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    up10 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    # up10 = Conv2DTranspose(L, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv1, up10], axis=3)
    conv11 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    final = Conv2D(1, 1)(conv11)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def C3_trained(inputs): ## Corregir pq se añaden 3 capas de convolucion mas y el L esta mal y los concatenate
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;
    L_7 = 7 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)



    up7 = Conv2D(L_6, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    # up7 = Conv2DTranspose(L_6, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv6, up7], axis=3) #conv3?
    conv8 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)



    up8 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv5, up8], axis=3)
    conv9 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    up9 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    up9 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv4, up9], axis=3)
    conv10 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv10)

    up10 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    up10 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv3, up10], axis=3)
    conv11 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    up11 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv11))
    # up11 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up11)
    merge12 = concatenate([conv2, up11], axis=3)
    conv12 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    up12 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv12))
    # up12 = Conv2DTranspose(L, [1, 2], activation='relu')(up12)
    merge13 = concatenate([conv1, up12], axis=3)
    conv13 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv13 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    final = Conv2D(1, 1, name='mapping')(conv13)

    return final

def C3(pretrained_weights=None, input_size=(256, 256, 31)):
    inputs = Input(input_size)
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;
    L_7 = 7 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)



    up7 = Conv2D(L_6, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    # up7 = Conv2DTranspose(L_6, [1, 2], activation='relu')(up7)
    merge7 = concatenate([conv6, up7], axis=3) #conv3?
    conv8 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv8 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)



    up8 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv5, up8], axis=3)
    conv9 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    up9 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    up9 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv4, up9], axis=3)
    conv10 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv10)

    up10 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    up10 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv3, up10], axis=3)
    conv11 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    up11 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv11))
    # up11 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up11)
    merge12 = concatenate([conv2, up11], axis=3)
    conv12 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv12 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    up12 = Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv12))
    # up12 = Conv2DTranspose(L, [1, 2], activation='relu')(up12)
    merge13 = concatenate([conv1, up12], axis=3)
    conv13 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv13 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    final = Conv2D(1, 1)(conv13)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def C4_trained(inputs): ## Corregir pq se añaden 3 capas de convolucion mas y el L esta mal y los concatenate
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;
    L_7 = 7 * L;
    L_8 = 8 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)
    conv8 = Conv2D(L_8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
    conv8 = Conv2D(L_8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)



    up8 = Conv2D(L_7, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L_7, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv7, up8], axis=3) #conv3?
    conv9 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)



    up9 = Conv2D(L_6, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    # up9 = Conv2DTranspose(L_6, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv6, up9], axis=3)
    conv10 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    up10 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    # up10 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv5, up10], axis=3)
    conv11 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv11)

    up11 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv11))
    # up11 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up11)
    merge11 = concatenate([conv4, up11], axis=3)
    conv12 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv12 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    up12 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv12))
    up12 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up12)
    merge12 = concatenate([conv3, up12], axis=3)
    conv13 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv13 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
    up13 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv13))
    up13 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up13)
    merge13 = concatenate([conv2, up13], axis=3)
    conv14 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv14 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    up14 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv14))
    # up14 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up14)
    merge14 = concatenate([conv1, up14], axis=3)
    conv15 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
    conv15 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)

    final = Conv2D(1, 1, name='mapping')(conv15)

    return final

def C4(pretrained_weights=None, input_size=(256, 256, 31)):
    inputs = Input(input_size)
    L = 16;
    L_2 = 2 * L;
    L_3 = 3 * L;
    L_4 = 4 * L;
    L_5 = 5 * L;
    L_6 = 6 * L;
    L_7 = 7 * L;
    L_8 = 8 * L;

    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    conv6 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    conv7 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)
    conv8 = Conv2D(L_8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
    conv8 = Conv2D(L_8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)



    up8 = Conv2D(L_7, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    # up8 = Conv2DTranspose(L_7, [1, 2], activation='relu')(up8)
    merge8 = concatenate([conv7, up8], axis=3) #conv3?
    conv9 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv9 = Conv2D(L_7, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)



    up9 = Conv2D(L_6, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    # up9 = Conv2DTranspose(L_6, [1, 2], activation='relu')(up9)
    merge9 = concatenate([conv6, up9], axis=3)
    conv10 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(L_6, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    up10 = Conv2D(L_5, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv10))
    # up10 = Conv2DTranspose(L_5, [1, 2], activation='relu')(up10)
    merge10 = concatenate([conv5, up10], axis=3)
    conv11 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv11 = Conv2D(L_5, 3, activation='relu', padding='same', kernel_initializer='he_normal', )(conv11)

    up11 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv11))
    # up11 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up11)
    merge11 = concatenate([conv4, up11], axis=3)
    conv12 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge11)
    conv12 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    up12 = Conv2D(L_3, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv12))
    up12 = Conv2DTranspose(L_3, [1, 2], activation='relu')(up12)
    merge12 = concatenate([conv3, up12], axis=3)
    conv13 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge12)
    conv13 = Conv2D(L_3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
    up13 = Conv2D(L_2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv13))
    up13 = Conv2DTranspose(L_2, [1, 2], activation='relu')(up13)
    merge13 = concatenate([conv2, up13], axis=3)
    conv14 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge13)
    conv14 = Conv2D(L_2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv14)
    up14 = Conv2D(L_4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv14))
    # up14 = Conv2DTranspose(L_4, [1, 2], activation='relu')(up14)
    merge14 = concatenate([conv1, up14], axis=3)
    conv15 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge14)
    conv15 = Conv2D(L_4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)

    final = Conv2D(1, 1)(conv15)

    model = Model(inputs, final)

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# PIX2PIX

# The facade training set consist of 400 images
BUFFER_SIZE = 30
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 30
# Each image is 256x256 in size
IMG_WIDTH = 124
IMG_HEIGHT = 128
OUTPUT_CHANNELS = 1
LAMBDA = 100

def downsample(filters, size, apply_batchnorm=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
