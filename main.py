from keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

# Define the input shape and number of neurons in the layers
input_shape = (100,)
latent_dim = 100
hidden_dim = 256

# Define the generator model
def build_generator():
    input_layer = Input(shape=input_shape)
    x = Dense(hidden_dim)(input_layer)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim*2)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim*4)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim*8)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dense(hidden_dim*4)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    output_layer = Dense(784, activation='sigmoid')(x)
    return Model(input_layer, output_layer)

# Define the discriminator model
def build_discriminator():
    input_layer = Input(shape=(784,))
    x = Dense(hidden_dim*4)(input_layer)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim*2)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    x = Dense(hidden_dim)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_layer)

# Build and compile the models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])

# Create the GAN
gan_input = Input(shape=input_shape)
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
