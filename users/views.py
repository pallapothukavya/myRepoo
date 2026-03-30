from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from .models import UserRegistrationModel
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model # type: ignore

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request,"Registration successful!")
    return render(request, 'UserRegistrations.html') 


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def index(request):
    return render(request,"index.html")

from django.shortcuts import render
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Activation, Dense, Dropout, Flatten, BatchNormalization,
                                     Conv2D, MaxPooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import itertools
import os
import warnings
import matplotlib.pyplot as plt
from .forms import ImageUploadForm
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)

def training(request):
    img_width, img_height = 120, 160
    train_data_dir = r'media\main_dataset\train'
    test_data_dir = r'media\main_dataset\test'
    nb_train_samples = 9957
    nb_validation_samples = 2487
    batch_size = 32
    epochs = 15

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_batches = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_batches = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=2487,
        class_mode='categorical'
    )

    # Model architecture
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_batches,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_batches,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4),
      
    ]
    )
    model.save('multiclass.h5')

    
    # Plot accuracy
  
    #Log loss score: 0.2251245001956247
   # Plot accuracy
    accs = history.history['accuracy']
    val_accs = history.history['val_accuracy']

    plt.plot(range(len(accs)), accs, label='Training Accuracy')
    plt.plot(range(len(val_accs)), val_accs, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    # Plot loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(range(len(loss)), loss, label='Training Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


    def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='',
                        cmap=plt.cm.Blues):
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        #plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm_plot_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='')

  
    


    return render(request, 'users/training.html',{'accuracy':accs, 'loss':loss, 'val_loss':val_loss})  # Replace with your actual template
class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predictions(request):
    predicted_class = None
    file_url = None
    model = load_model('multiclass.h5')

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)

        img_path = os.path.join(fs.location, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 120))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)

        threshold = 0.98

        if confidence < threshold:
            predicted_class = f"This is not a valid White Blood Cell image (Confidence: {confidence:.2f})"
        else:
            predicted_class = f"{class_names[predicted_index]} (Confidence: {confidence:.2f})"

    return render(request, 'users/detection.html', {
        'predicted_class': predicted_class,
        'image_url': file_url
    })

@csrf_exempt
def api_predictions(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)

        img_path = os.path.join(fs.location, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 120))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model('multiclass.h5')
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))

        threshold = 0.98
        if confidence < threshold:
            result = "This is not a valid White Blood Cell image"
            status = "invalid"
        else:
            result = class_names[predicted_index]
            status = "success"

        return JsonResponse({
            'status': status,
            'predicted_class': result,
            'confidence': confidence,
            'image_url': request.build_absolute_uri(file_url)
        })
    return JsonResponse({'status': 'error', 'message': 'No image uploaded'}, status=400)

@csrf_exempt
def api_user_login(request):
    if request.method == "POST":
        import json
        try:
            data = json.loads(request.body)
            loginid = data.get('loginid')
            pswd = data.get('password')
        except:
            loginid = request.POST.get('loginid')
            pswd = request.POST.get('password')

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                return JsonResponse({
                    'status': 'success',
                    'message': 'Login successful',
                    'user': {
                        'id': check.id,
                        'name': check.name,
                        'email': check.email
                    }
                })
            else:
                return JsonResponse({'status': 'error', 'message': 'Account not activated'}, status=401)
        except UserRegistrationModel.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Invalid Login ID or Password'}, status=401)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

@csrf_exempt
def api_user_register(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        try:
            user = UserRegistrationModel(
                name=data.get('name'),
                loginid=data.get('loginid'),
                password=data.get('password'),
                mobile=data.get('mobile'),
                email=data.get('email'),
                locality=data.get('locality', ''),
                address=data.get('address', ''),
                city=data.get('city', ''),
                state=data.get('state', ''),
                status='waiting'
            )
            user.save()
            return JsonResponse({'status': 'success', 'message': 'Registration successful! Waiting for admin activation.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)
