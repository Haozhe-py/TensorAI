import tensorflow.keras as k
import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os

def read_png(fpath:str) -> np.array:
    try:
        if fpath[0] == '\"' and fpath[-1] == '\"':
            fpath = fpath[1:-1]
        elif fpath[0] == '\'' and fpath[-1] == '\'':
            fpath = fpath[1:-1]
        img = Image.open(fpath).resize((28,28)).convert('L')
        result = np.array(img)/255.0
        result = 1.0-result
        result = np.array([result.tolist()])
        return result
    except:
        print('ERROR: Cannot load file',file=sys.stderr)

def train() -> None:
    mnist = k.datasets.mnist
    (xtrain,ytrain),(xtest,ytest)=mnist.load_data()
    xtrain, xtest = xtrain/255.0, xtest/255.0
    xtrain, xval = xtrain[:50000],xtrain[50000:]
    ytrain, yval = ytrain[:50000],ytrain[50000:]
    model = k.models.Sequential([
        k.layers.Conv2D(1,(3,3),padding='valid',activation='relu',input_shape=(28,28,1)),
        k.layers.Conv2D(1,(3,3),padding='valid',activation='relu'),
        k.layers.MaxPooling2D(pool_size=(2,2)),
        k.layers.Flatten() ,
        k.layers.Dense(256,activation='relu') ,
        k.layers.Dropout(0.2) ,
        k.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    callbacks = [
        k.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            mode='max'
        ) ,
        k.callbacks.ModelCheckpoint(
            filepath='trained_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    model.fit(xtrain,ytrain,epochs=25,callbacks=callbacks,validation_data=(xval,yval))
    del model
    model = k.models.load_model('trained_model.keras')
    print('\n\n\n\nEvaluate:\n',model.evaluate(xtest,ytest))
    return None

def run() -> None:
    try:
        model = k.models.load_model('trained_model.keras')
    except:
        print('ERROR: Model not found!\nTraining TensorAI...',file=sys.stderr)
        train()
        run()
    if os.path.join('a','b')=='a\\b':               #check if the system is Windows
        os.system('cls')
    else:
        os.system('clear')                          #for Linux
    print('欢迎使用TensorAI-0.0.0，此程序可用于识别手写阿拉伯数字。')
    while True:
        try:
            f = input('请输入PNG图片路径：（输入exit退出）\n')
            if f == 'exit':
                break
            img = read_png(f)
            pred = tf.argmax(model(img),axis=-1).numpy().tolist()[0]
            print('识别结果：',pred)
            print('\n\n')
            continue
        except:
            print('ERROR',file=sys.stderr)
    return None

def h() -> None:
    print('Usage:')
    print('python3 main.py train  ............ Train the model')
    print('python3 main.py run  .............. Run    TensorAI')
    return None
def main(args) -> int:
    if len(args) == 0:
        main(['run'])
    elif len(args) == 1:
        if args[0]=='train':
            train()
        elif args[0]=='run':
            run()
        elif args[0] in ['-?','/?','--help','-help','-h','/help','/h']:
            h()
        else:
            print('ERROR: Invalid syntax\n\n',file=sys.stderr)
            h()
    else:
        print('ERROR: Invalid syntax\n\n',file=sys.stderr)
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])