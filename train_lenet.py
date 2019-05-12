import os
import argparse
import tensorflow as tf


def _parse_args():
    parser = argparse.ArgumentParser(description='Train LeNet')
    parser.add_argument('--output_dir', required=False, default="model", type=str)
    return parser.parse_args()


def main():
    output_dir = _parse_args().output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # TODO(Hongting Wang): After implementing the convolution layer, add some convolution layers in the model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    with open(os.path.join(output_dir, 'lenet_model.json'), 'w') as f:
        f.write(model.to_json())
    with open(os.path.join(output_dir, 'lenet_weights.bin'), 'wb') as f:
        for w in model.get_weights():
            f.write(w[-1].tobytes())


if __name__ == "__main__":
    main()
