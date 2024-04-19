import os
import cv2
import numpy as np

data_directory = os.getcwd() + '/Dataset/'
domains = ['amazon', 'dslr', 'webcam']
classes = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle',
           'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet',
           'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone',
           'monitor', 'mouse', 'mug', 'paper_notebook', 'pen',
           'phone', 'printer', 'projector', 'punchers', 'ring_binder',
           'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser',
           'trash_can']

if not os.path.exists('data'):
    os.makedirs('data')

for domain in domains:
    domain_path = os.path.join(data_directory, domain)
    domain_data = []
    domain_label = []
    label = 0
    for obj in classes:
        obj_dir = domain_path + '/' + obj
        for file in os.listdir(obj_dir):
            src = obj_dir + '/' + file
            img = cv2.imread(src)
            resize_img = cv2.resize(img, (224, 224))
            domain_data.append(resize_img)
            domain_label.append(label)
        label += 1
    np.save(f'data/{domain}_data', domain_data)
    np.save(f'data/{domain}_label', domain_label)