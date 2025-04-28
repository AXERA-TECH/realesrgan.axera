import cv2
import numpy as np
import pyaxengine.axengine as ort


def get_model() -> ort.InferenceSession:
    model = ort.InferenceSession("models/x4-256.axmodel")

    for input in model.get_inputs():
        print(input.name, input.shape, input.dtype)

    for output in model.get_outputs():
        print(output.name, output.shape, output.dtype)
    width = model.get_inputs()[0].shape[2]
    height = model.get_inputs()[0].shape[1]
    
    return model, width, height

    
def preprocess_image(image, width=64, height=64):
    data = cv2.resize(image, (width, height))
    data = np.expand_dims(data, axis=0)
    return data


if __name__ == "__main__":
    model, width, height = get_model()
    
    img = cv2.imread("./input.png")
    
    print(img.shape)
    img = preprocess_image(img, width, height)
    print(img.shape)
    output = model.run(None, {"input.1": img})[0]
    print(output.shape)
    
    output[output>1] = 1
    output[output<0] = 0
    output_img = (output * 255).astype(np.uint8)[0]
    
    
    print(output_img.shape)
    cv2.imwrite("./output.png", output_img)