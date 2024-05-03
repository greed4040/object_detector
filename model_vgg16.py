from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape, num_classes):
    # Загрузка предобученной VGG16 без верхних слоев
    backbone = keras.applications.VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Заморозка предобученных слоев
    for layer in backbone.layers:
        layer.trainable = False
    
    # Дополнительные сверточные слои
    x = backbone.output
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Адаптивный глобальный пулинг
    x = layers.GlobalAveragePooling2D()(x)
    
    # Голова классификации
    class_head = layers.Dense(512, activation='relu')(x)
    class_head = layers.Dense(num_classes, activation='softmax', name='class')(class_head)
    
    # Голова обнаружения (координаты)
    bbox_head = layers.Dense(512, activation='relu')(x)
    bbox_head = layers.Dense(4, name='bbox')(bbox_head)
    
    # Создание модели
    model = keras.Model(
        inputs=backbone.input, 
        outputs=[class_head, bbox_head]
    )
    
    return model

# Создание и компиляция модели
input_shape = (None, None, 3)  # Гибкое входное разрешение
num_classes = 10
model = create_model(input_shape, num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'class': 'categorical_crossentropy',
        'bbox': 'mse'
    },
    loss_weights={
        'class': 1.0,
        'bbox': 1.0
    },
    metrics={'class': 'accuracy'}
)

# Печать архитектуры модели
model.summary()
