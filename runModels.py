# This file exists solely of the ability to test the models, whether they be an ONNX of h5 model and see what the output would be.
from keras.preprocessing.image import load_img, img_to_array
IMG_DIM = (350, 350)
input_shape = (350, 350, 3)

# translator is a simple dict based on the possible outputs of your model, can be found during model generation.
translator = {'0': 'americanMoose', '1': 'canadaLynx', '2': 'cougar', '3': 'grizzlyBear', '4': 'northernBaldEagle', '5':'raccoon', '6': 'trumpeterSwan', '7': 'woodBison'}

test_img = img_to_array(load_img("83727230_171197127476202_2407213256830615552_n.jpg", target_size=IMG_DIM))

# reshape the image
test_img = test_img.reshape(1, 350, 350, 3)

# normalize image
test_img_scaled = test_img.astype('float32')
test_img_scaled /= 255


#THIS BLOCK IS USED SOLELY TO TEST MODEL WITH h5/original to run against

from keras.models import load_model
model = load_model("simple_zoo_imageAug_with_Transfer_learning2_FullRun.h5")

# predict digit
prediction = model.predict(test_img_scaled)
print(prediction)
print(prediction.argmax())
print(translator[str(prediction.argmax())])

#This block will be used to test as onnx

import onnxruntime as rt

sess = rt.InferenceSession("simple_zoo_imageAug_with_Transfer_learning2_FullRun.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: test_img_scaled})[0]
print(pred_onx)
print(pred_onx.argmax())
print(translator[str(pred_onx.argmax())])
