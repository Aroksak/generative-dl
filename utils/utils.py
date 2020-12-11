import numpy as np
from PIL import Image
from IPython.display import display
from IPython.display import Image as ImageDisplay
from io import BytesIO


def output_to_pil(output_tensor):
    x = output_tensor.squeeze().cpu().detach().numpy()
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return Image.fromarray(x.astype(np.uint8), mode='L')


def display_img(im):
    bio = BytesIO()
    im.save(bio, format='png')
    display(ImageDisplay(bio.getvalue(), format='png'))
