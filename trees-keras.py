import os
import argparse

from PIL import Image
import numpy as np

CAT = ['榕樹', '白千層', '楓香', '台灣欒樹', '小葉欖仁', '大葉欖仁', '茄冬',
       '黑板樹', '大王椰子', '鳳凰木', '阿勃勒', '水黃皮', '樟樹', '苦楝']

def preprocess_image(f, res=200, expand=False, precision='float32'):
    img = Image.open(f)
    img = img.resize((res, res))
    data = np.array(img.getdata())
    img = data.reshape(1, *img.size, 3) if expand else data.reshape(*img.size, 3)
    img = img.astype(precision)
    img /= 255.
    
    return img
    
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('tree', type=str)
args = parser.parse_args()

# inference for Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model    
model = load_model(args.model)
img = preprocess_image(args.tree, expand=True)
pred = model.predict(img)

print(f'{CAT[np.argmax(pred[0])]}')
