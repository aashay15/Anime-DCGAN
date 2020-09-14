import glob
import os
from PIL import Image

for index, filename in
    enumerate(glob.glob('/path/to/directory/containing/images/*.*')):
    try:
    # Open image
    im = Image.open(filename)
    except Exception as e:
        print("Exception:{}".format(e))
        continue

    cropped_image = im.resize((64, 64), Image.ANTIALIAS)
    cropped_image.save("/path/to/directory/to/store/cropped/images/filename.png"))
