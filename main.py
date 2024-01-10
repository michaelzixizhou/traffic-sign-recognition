# load and evaluate a saved model
import numpy as np
import pandas as pd
from keras.models import load_model, Model

from model import preprocessing
import cv2

def get_label(class_id, df):
    try:
        # Use .loc to retrieve the corresponding text for the given ClassId
        text = df.loc[df['ClassId'] == class_id, 'Name'].values[0]
        return text
    except IndexError:
        # Handle the case where the ClassId is not found
        return "ClassId not found"


if __name__ == "__main__":
    # load model
    model = load_model("cnnmodel.h5")
    # summarize model.
    if isinstance(model, Model):
        path = r"./testcases/speed-limit-traffic-sign-80-vector-24456601.jpg"
        labelfile = "./archive/labels.csv"
        img = preprocessing(cv2.imread(path))
        
        cv2.imshow("sign", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        data = pd.read_csv(labelfile)

        
        lst = np.array([img])
        lst.reshape(1, 100, 100, 1)
        predictions = model.predict(lst)

        print(get_label(np.argmax(predictions), data))
