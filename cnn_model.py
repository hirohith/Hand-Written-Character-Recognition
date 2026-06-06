from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

# Convolution Layer 1
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

# Pooling Layer
model.add(
    MaxPooling2D(pool_size=(2, 2))
)

# Flatten Layer
model.add(Flatten())

# Hidden Layer
model.add(
    Dense(
        128,
        activation='relu'
    )
)

# Output Layer
model.add(
    Dense(
        10,
        activation='softmax'
    )
)

model.summary()
