from scipy.ndimage import morphology
from sklearn.metrics import confusion_matrix
import glob
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


def surfd(input1, input2, sampling=1, connectivity=1):

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds


truth_fnames = list(glob.glob("./mri_data/labeled/2017/truth/*.png"))
predict_fnames = list(glob.glob("./mri_data/labeled/2017/label_/*.png"))

truth_numbers = []
predict_numbers = []


for t_f in truth_fnames:
    truth_numbers.append(re.findall(r'\d+', t_f)[-1])

for p_f in predict_fnames:
    predict_numbers.append(re.findall(r'\d+', p_f)[-1])

truth_numbers = dict(zip(truth_fnames, truth_numbers))
predict_numbers = dict(zip(predict_fnames, predict_numbers))

truth_fnames.sort(key=lambda fname: int(truth_numbers[fname]))
predict_fnames.sort(key=lambda fname: int(predict_numbers[fname]))

truth = []
predict = []

for t_f in truth_fnames:
    truth.append(np.array(Image.open(t_f).resize((256, 256))))

for p_f in predict_fnames:
    predict.append(np.array(Image.open(p_f).resize((256, 256))))

truth = np.array(truth)
predict = np.array(predict)

num_pixels = truth.shape[0] * truth.shape[1] * truth.shape[2]

truth = (truth > 0.5).astype('uint8')
predict = (predict > 0.5).astype('uint8')

truth_f = truth.reshape(-1, 1)
predict_f = predict.reshape(-1, 1)
confusion = np.zeros((2, 2))


tn, fp, fn, tp = confusion_matrix(truth_f, predict_f).ravel()
confusion = np.array([[tp, fp], [fn, tn]])
print(confusion)
#confusion = confusion / confusion.sum()

dice = (2 * tp) / (2*tp + fp + fn)
gce = min([fn*(fn+2*tp)/(fp+fn)+fp*(fp+2*tn)/(tn+fp),
           fp*(fp+2*tp)/(tp+fp) + fn*(fn+2*tn)/(tn+fn)]) / num_pixels

vs = 1 - (abs(fn - fp) / (2*tp + fp + fn))

a = 0.5*(tp*(tp-1) + fp*(fp-1)+tn*(tn-1)+fn*(fn-1))
b = 0.5*((tp+fn)**2 + (tn+fp)**2 - (tp**2+tn**2+fp**2+fn**2))
c = 0.5*((tp+fp)**2 + (tn+fn)**2 - (tp**2+tn**2+fp**2+fn**2))
d = num_pixels * (num_pixels - 1) / 2 - (a + b + c)

ri = (a+b) / (a+b+c+d)

""" 
cmap = plt.cm.Blues
classes = ['Colon', 'Non-Colon']
title = 'Confusion Matrix'
fig, ax = plt.subplots()
im = ax.imshow(confusion, cmap=cmap)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(confusion.shape[1]),
       yticks=np.arange(confusion.shape[0]),
       xticklabels=classes,
       yticklabels=classes,
       title=title,
       ylabel='Truth',
       xlabel='Prediction')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(j, i, format(confusion[i, j], 'f'),
                ha='center', va='center', color='white')
plt.show()
"""

truth_pos = np.swapaxes(np.vstack(np.where(truth == 1)), 0, 1)
predict_pos = np.swapaxes(np.vstack(np.where(predict == 1)), 0, 1)

surface_distance = surfd(predict, truth, [1.25, 1.25, 10], 1)
print(f'msd: {surface_distance.mean()}')
print(f'hd: {surface_distance.max()}')
""" print(truth_pos.shape)
print(predict_pos.shape)
print(directed_hausdorff(truth_pos, predict_pos))
print(directed_hausdorff(predict_pos, truth_pos))
 """
print(f'dice: {dice}')
print(f'gce: {gce}')
print(f'vs: {vs}')
print(f'ri: {ri}')
