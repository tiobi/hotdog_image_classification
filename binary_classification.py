# import tools
import numpy as np
from tqdm import tqdm
import keras
from keras import layers
#from keras.callbacks import TensorBoard as tb #텐서보드 만들고 활성화


'''
TO DO:
    과적합 방지, CALLBACK
    이미지 샘플 나타내기
    
    
ON PRG:
    텐서보드:
        html에서만 되는 것 같음. 
        새로운 페이지에너 어떻게 나타내는지 모르겠음.
        line magic을 스파이더에서 어떻게 쓰는지 알아봐야 함.
        
    dir return 합수:
        테스트/트레인, 카테고리별로 어떻게 반환해야 하는지 고민중.
        
DONE: 
    기능들 함수화:
        모델 만들기(v), 
        폴더 지정(x)
    모델 저장(필요없음)
    적은 데이터로 학습이미지 데이터 생성
'''
 

CATEGORIES = ["hotdog", "not-hotdog"]
IMG_SIZE = 100


#디렉터리 반환 (진행중)
'''
def get_dir():
    directory = []
    for category in CATEGORIES:
        directory.append(os.path.join(r".\\", category))
    
    return directory
'''    

'''
TensorBoard Activation:
    import IPython
    IPython.start_ipython
'''    
#이미지 데이터 생성
def generate_images():
    from keras.preprocessing.image import ImageDataGenerator
    from keras.preprocessing.image import array_to_img, img_to_array, load_img
    import os
    
    #반복 횟수
    TARGET = 1
    datagen = ImageDataGenerator(
        rotation_range = 50,
        width_shift_range = 0.2, 
        height_shift_range = 0.2, 
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest")
    
    for category in CATEGORIES:
        PATH = os.path.join(r".\train", category)
        for img in tqdm(os.listdir(PATH)): #dir return 함수 필요함
            image = load_img(os.path.join(PATH, img))
            x = img_to_array(image)
            x = x.reshape((1, ) + x.shape)

            i = 0
            for batch in datagen.flow(x, 
                                      batch_size = 1, 
                                      save_to_dir = PATH,
                                      save_prefix = "gen",
                                      save_format = "png"):
                i += 1
                if i == TARGET:
                    break
                

#트레이닝 이미지 리스트 생성 함수
def create_training_data():
    import os
    import cv2
    import time
    
    training_data = []
    DATADIR = os.path.join(r".\\", "train")
    print("Creating Training Data")
    time.sleep(2)
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                
            except Exception as e:
                pass
            
    return training_data
 
    
#트레이닝 이미지 텐서 생성    
def create_training_tensor():               

    training_data = create_training_data()
    
    #트레이닝 데이터 랜덤화
    import random
    random.shuffle(training_data)
    
    #케라스 입력 형식 변환
    x_train = []
    y_train = []
    
    for features, label in training_data:
        x_train.append(features)
        y_train.append(label)
             
    x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)/255.
    y_train = np.array(y_train)   
    return x_train, y_train


#테스트 이미지 생성 함수
def create_testing_data():
    import os
    import cv2
    import time
    
    DATADIR_TEST = os.path.join(r".\\", "test")
    testing_data = []
    print("Creating Testing Data")
    time.sleep(2)
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR_TEST, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
                
            except Exception as e:
                pass
            
    return testing_data


#테스트 이미지 텐서 생성
def create_testing_tensor():
   
    testing_data = create_testing_data()
    
    #테스트 데이터 랜덤
    import random
    random.shuffle(testing_data)
    
    #입력변환
    x_test = []
    y_test = []
    
    for features, label in testing_data:
        x_test.append(features)
        y_test.append(label)
     
    x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.
    y_test = np.array(y_test)   
    return x_test, y_test


#데이터셋 로드
def load_dataset():
    x_train, y_train = create_training_tensor()
    x_test, y_test = create_testing_tensor()    
    return x_train, y_train, x_test, y_test


#모델 로드
def get_model(IMG_SIZE):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(256, (3,3), activation = "relu"))
    model.add(layers.MaxPooling2D(pool_size = (2, 2)))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))
    return model

    
#학습 그래프 시각화
def show_graphs(history):
    import matplotlib.pyplot as plt
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) +1)
    plt.plot(epochs, acc, label = "Training Accuracy")
    plt.plot(epochs, val_acc, label = "Validation Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, label = "Training Loss")
    plt.plot(epochs, val_loss, label = "Validation Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    
    plt.show()
    plt.savefig("1000_images.png", dpi = 1000)

'''
x_train, y_train, x_test, y_test = load_dataset()
model = get_model(IMG_SIZE)
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
hist = model.fit(x_train, y_train, batch_size = 25, epochs = 150, validation_data = (x_test, y_test))
show_graphs(hist)
'''















   