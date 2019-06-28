import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

emotion_strings = [
    'Angry'      # 0
    , 'Disgust'  # 1
    , 'Fear'     # 2
    , 'Happy'    # 3
    , 'Sad'      # 4
    , 'Surprise' # 5
    , 'Neutral'  # 6
]

def load_fer_dataset(path):
    width = 48
    height = 48

    data = pd.read_csv(path)
    # Covert pixels column to a list
    pixels = data['pixels'].tolist()
    emotions = pd.get_dummies(data['emotion']).values

    faces = []

    for pixel_sequence in pixels:
        # Split the string by space character as a list
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        
        # convert to numpy array
        face = np.asarray(face).reshape(width, height)        
        faces.append(face.astype('float32'))
    # Convert to numpy array   
    faces = np.asarray(faces)
    # normalize data 
    faces /= 255.0
    # center data
    faces = faces - np.mean(faces)
    # Expand to (48, 48, 1)
    faces = np.expand_dims(faces, -1)

    print("Loaded emotions array with shape {}".format(emotions.shape))
    print("Loaded image array with shape {}".format(faces.shape))
    return faces, emotions

def show_random_sample(faces, emotions):
    r = np.random.randint(len(faces), size=10)
    for i in r:
        img = faces[i,:,:,0]
        plt.imshow(img, cmap='gray')
        plt.title(emo_to_string(emotions[i]))
        plt.show()

def split_dataset(faces, emotions, train_size=28709, validation_size=3589, test_size=3589):
    perm = np.random.permutation(faces.shape[0])
    train_dataset = faces[perm[:train_size]]
    train_labels = emotions[perm[:train_size]]
    valid_dataset = faces[perm[train_size:train_size+validation_size]]
    valid_labels = emotions[perm[train_size:train_size+validation_size]]
    test_dataset = faces[perm[train_size+validation_size:]]
    test_labels = emotions[perm[train_size+validation_size:]]

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def emo_to_string(emotion):
    return emotion_strings[np.argmax(emotion)]

def show_data_distribution(emotions):
    # obtain counts for each of the categories
    counts = collections.Counter([np.argmax(row) for row in emotions])
    emo_ids = list(counts.keys())
    emo_counts = list(list(counts.values()))
    bars = plt.bar(emo_ids, emo_counts, color='g')
    plt.xticks(emo_ids, [emotion_strings[i] for i in emo_ids])
    plt.xlabel("Emotion category")
    plt.ylabel("Number of examples")
    for i, bar in enumerate(bars):
        y = bar.get_height()
        plt.text(bar.get_x() + 0.1, y + 1, emo_counts[i])
    plt.show()


