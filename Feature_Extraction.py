import tensorflow as tf
import cv2
import os

# TFRecord files
files = {
    "train": "train_vehicles.tfrecord",
    "val": "val_vehicles.tfrecord",
    "test": "test_vehicles.tfrecord"
}

# Create dataset folders
for split in files:
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)

# Describe TFRecord structure
feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),

    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),

    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}

def parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def convert_tfrecord(tfrecord_path, split):

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)

    for record in dataset:

        filename = record['image/filename'].numpy().decode()
        image = tf.image.decode_jpeg(record['image/encoded']).numpy()

        xmin = tf.sparse.to_dense(record['image/object/bbox/xmin']).numpy()
        xmax = tf.sparse.to_dense(record['image/object/bbox/xmax']).numpy()
        ymin = tf.sparse.to_dense(record['image/object/bbox/ymin']).numpy()
        ymax = tf.sparse.to_dense(record['image/object/bbox/ymax']).numpy()
        labels = tf.sparse.to_dense(record['image/object/class/label']).numpy()

        # Save image
        cv2.imwrite(f"dataset/images/{split}/{filename}", image)

        label_file = filename.replace(".jpg", ".txt")

        with open(f"dataset/labels/{split}/{label_file}", "w") as f:

            for i in range(len(labels)):

                x_center = (xmin[i] + xmax[i]) / 2
                y_center = (ymin[i] + ymax[i]) / 2
                width = xmax[i] - xmin[i]
                height = ymax[i] - ymin[i]

                class_id = labels[i] - 1

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Convert all datasets
for split, path in files.items():
    print("Processing:", split)
    convert_tfrecord(path, split)

print("Dataset conversion complete.")