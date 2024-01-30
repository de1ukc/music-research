import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from msd_script import collect_dataset, msd_subset_path, plot_tempogram, save_images, plot_chromagram, plot_beatgram, \
    plot_spec, N
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Загрузка данных

beats, tempos, segments_pitches_list, mfccs_list = collect_dataset(basedir=msd_subset_path)


# save_images(func=plot_tempogram, array=beats, filepath='imgs/tempograms', gramm_name='tempogram')
# save_images(func=plot_chromagram, array=segments_pitches_list, filepath='imgs/chromagrams', gramm_name='chromagrams')


# save_images(func=plot_beatgram, array=beats, filepath='imgs/beatgrams', gramm_name='beatgrams')
# save_images(func=plot_spec, array=mfccs_list, filepath='imgs/specs', gramm_name='specs')


# Функция загрузки данных
def load_data(path, num_samples, gram_name, target_size=(1000, 600), bpm_values=tempos):
    graphs = []
    bpms = []
    for i in range(0, num_samples - 1):
        # Загружаем изображение графика и изменяем его размер
        img = load_img(f"{path}/{gram_name}_{i}.png", color_mode='grayscale', target_size=target_size)
        img_array = img_to_array(img)
        graphs.append(img_array)

        # Загружаем соответствующее BPM значение
        bpm = bpm_values[i]
        bpms.append(bpm)
    return np.array(graphs), np.array(bpms)


# Создание модели
def create_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        Dropout(0.5),  # Добавляем слой Dropout с коэффициентом отсева 0.5
        layers.Dense(1, name='output')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Обучение модели и построение кривой обучения
def train_and_plot_learning_curve(path, gram_name, epochs, num_samples=N):
    # Загрузка данных
    graphs, bpms = load_data(path=path, num_samples=num_samples, gram_name=gram_name)
    graphs = graphs / 255.0  # Нормализация значений пикселей

    # Разделение данных на обучающий и тестовый наборы
    train_graphs, test_graphs, train_bpms, test_bpms = train_test_split(graphs, bpms, test_size=0.2, random_state=42)

    # Создание модели
    model = create_model(input_shape=(1000, 600, 1))

    train_and_evaluate_learning_curve(model, epochs, train_graphs, train_bpms, test_graphs, test_bpms)


# Обучение модели и оценка кривой обучения
def train_and_evaluate_learning_curve(model, epochs, train_graphs, train_bpms, test_graphs, test_bpms):
    # Списки для хранения значений метрики
    training_accuracies = []
    validation_accuracies = []

    # Обучение модели
    for epoch in range(epochs):
        # Обучение на одной эпохе
        model.fit(train_graphs, train_bpms, epochs=1, batch_size=32, validation_data=(test_graphs, test_bpms))

        # Оценка производительности модели на обучающем и валидационном наборах данных
        train_loss, train_accuracy = model.evaluate(train_graphs, train_bpms, verbose=0)
        validation_loss, validation_accuracy = model.evaluate(test_graphs, test_bpms, verbose=0)

        # Сохранение значений точности
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        # Вывод информации о текущей эпохе
        print(
            f'Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy:.4f} - Validation Accuracy: {validation_accuracy:.4f}')

    # Построение кривой обучения на основе точности
    plt.plot(range(1, epochs + 1), training_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve (Accuracy)')
    plt.legend()
    plt.show()


# Обучение модели на 10 эпохах и построение кривой обучения
# train_and_plot_learning_curve(path='imgs/tempograms', gram_name='tempogram', epochs=10)
train_and_plot_learning_curve(path='imgs/chromagrams', gram_name='chromagrams', epochs=10)
