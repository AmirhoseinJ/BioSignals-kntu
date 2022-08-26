import os.path

from PIL import Image

path = r"C:\****\ditt\img"
dirs = os.listdir(path)
global i


def crop():
    i = 0
    for item in dirs:
        fullpath = os.path.join(path, item)  # corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, ext = os.path.splitext(fullpath)
            imCrop = im.crop((0, 0, 1920, 1040))  # corrected
            imCrop.save(f[:-4] + '\crp\Cropped' + str(i) + '.png', "PNG", quality=100)
            i += 1


crop()

im = Image.open(r"C:\****\ditt\img\p_5.png")
imCrop = im.crop((0, 0, 1920, 1040))
imCrop.show()
print(imCrop)
imCrop.save(r"C:\****\ditt\img\crp\p_5.png" + 'Cropped.png', "PNG", quality=100)
print(os.path.splitext(r"C:\****\ditt\img\p_5.png"))
