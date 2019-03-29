from util import data_io, data_process


from PIL import Image


dicom_nps = data_io.preprocessDicom('./mri_data/labeled/2017/truth2')
norm_dicom_nps = []

for dicom_np in dicom_nps:
    norm_dicom_nps.append((data_process.normalize(dicom_np)*255).astype('uint8'))


for i, norm_np in enumerate(norm_dicom_nps):
    Image.fromarray(norm_np).save(f'./mri_data/labeled/2017/truth2_png/seg_{i}.png')
