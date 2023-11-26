# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:43:44 2022

@author: Abhishek Shankar
"""


import os
from PIL import Image
import cv2 
from IPython.display import display
import numpy as np
import pandas as pd
from fpdf import FPDF
from PIL import Image as pili, ImageDraw as pild, ImageFont as pilf, ImageOps as piliops

TINT_COLOR = (0, 0, 0)  # Black
OPACITY = int(255 * .50)
FONT = pilf.truetype("Inkfree.ttf", 24) # Font
IMG_BASE_WIDTH = 600
IMG_NUMBERS = 47

def convert_from_cv2_to_image(img: np.ndarray) -> pili:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #return pili.fromarray(img)

def convert_from_image_to_cv2(img: pili) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #return np.asarray(img)
    
# Controls blur amount and line size and number of text lines. No bold font!

def imgcompress_mem(path_in, k):
    img = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)
#    img = np.rot90(img,axes=(-2,-1))
    # set the ratio of resized image
    width = int((img.shape[1])/k)
    height = int((img.shape[0])/k)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = np.rot90(img,3)
    # resize the image by resize() function of openCV library
    return img


def cartoonizeblt_mem_nb(path_in, k, blur, line, text, nlines=1, font='verdana'):
    
    #print(path_in)
    imgc = imgcompress_mem(path_in, k)

    line_size = line
    blur_value = blur
    gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    bigedges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    bigedges_pil = cv2.cvtColor(bigedges, cv2.COLOR_BGR2RGB) # Converting BGR to RGB

    toon = cv2.bitwise_and(imgc, imgc, mask=bigedges)
    if 0 == len(text):
        return toon
    
    print('Using font ' + font + '...')
    myfont = (
        pilf.truetype("ITCKRIST.TTF", 
            24 if k == 16 else 18 if k == 14 else 18 if k == 12 else 20 if k == 8 else 82) if font=='ITCKRIST'
        else
            pilf.truetype("Inkfree.ttf", 
                24 if k == 16 else 18 if k == 14 else 18 if k == 12 else 20 if k == 8 else 82) if font=='Inkfree'
        else
            pilf.truetype(font + ".ttf", 24 if k == 16 else 18 if k == 14 else 18 if k == 12 else 20 if k == 8 else 82)
    )


    cblimg_pil = Image.fromarray(cv2.cvtColor(toon, cv2.COLOR_BGR2RGBA))

    overlay = pili.new('RGBA', cblimg_pil.size, TINT_COLOR+(0,))
    draw = pild.Draw(overlay)
    #_, h = FONT.getsize(text)
    _, h = myfont.getsize(text)
    num_lines = nlines
    x, y = 0, cblimg_pil.height - (num_lines)*h-10
    draw.rectangle((x, y, x + cblimg_pil.width, y + (num_lines)*h+10), fill=TINT_COLOR+(OPACITY,))
    if k == 1:
        draw.text((x+10, y), text, fill=(248,248,248), font=myfont) #, stroke_width=1)
    elif k < 8:
        draw.text((x+10, y), text, fill=(248,248,248), font=myfont)
    else:
        draw.text((x+10, y), text, fill=(248,248,248), font=myfont) #, stroke_width=1)

    cblimg_pil = pili.alpha_composite(cblimg_pil, overlay)
    cblimg_pil = cblimg_pil.convert("RGB")

    return convert_from_image_to_cv2(cblimg_pil)


#wd = r"C:\Users\abhis\Desktop\NEU\INFO6105 Data Sci Eng Methods\Assignment 5\\"
#
#cbltimg = cartoonizeblt_mem_nb(wd+'toons/dino/P1250231.jpg', 4, 9, 11, 'hello \nworld', nlines=2, 
#                            font='arial')
#cbltimg_pil = cv2.cvtColor(cbltimg, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
#display(pili.fromarray(cbltimg_pil))
#cbltimg_pil.shape


# # Producing a storybook strip
# Here's an example of how to use the `zipper` and **list comprehensions** in order to work with, not a single image, but *many images at the same time*! I told you that is what programming is all about, so we are programming a lot here below!



def simple_row(folder, list_im, list_opt, list_txt, list_nlines, font='arial'):
    
    # cartoonize them in memory with text
    cimgs = [ cartoonizeblt_mem_nb(folder + '/' + i + '.jpeg', 14, 3, 3, k, nlines=n, font=font)
              for i,j,k,n in zip(list_im, list_opt, list_txt, list_nlines) ]
    print(len(cimgs))
    #print(cimgs)
    # resize
    heighto = int(cimgs[0].shape[0])
    widtho = int(cimgs[0].shape[1])
    # heighto / widtho = height / width ==> height = heighto / widtho * width
    width = 245
    height = int(heighto / widtho * width)
    cimgs_dim = (width, height)
    cimgsr = [ cv2.resize(cimgs[i], cimgs_dim, interpolation = cv2.INTER_AREA) for i in range(len(list_im))]
    print(len(cimgsr))
    # add borders
    white = [255,255,255]
    bcimgs = [ cv2.copyMakeBorder(i, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=white) for i in cimgsr ]
#    print(bcimgs)
#    print(list_im)
    print(len(bcimgs))
    print(len(list_im))
    # stack them horizontally
    return np.concatenate([ bcimgs[i] for i in range(0,len(list_im)) ], axis=1)


# Here I create a horizontal strip from pictures from my bike trip:

# In[ ]:



#wd_pics = r"C:/Users/abhis/Desktop/NEU/INFO6105 Data Sci Eng Methods/Assignment_5_Group3_Abhishek_Anvi_Shobith/Assignment 5 data science -20221013T154820Z-001/Assignment 5 data science"
##wd_pics = r"C:\Users\abhis\Desktop\Pics\\"
#
#grp_pics = [files.split(".")[0] for files in os.listdir(wd_pics) if files.endswith(".jpeg")][0:4]
#lines = ["BLAH BLAH" for x in grp_pics][0:4]
#grp_opt = ['pencil' for x in grp_pics][0:4]
#txt_nlines = [1 for x in grp_pics][0:4]
##rows4 = simple_row (
##          wd_pics, 
##          grp_pics,
##         ['pencil', 'pencil', 'buildings', 'buildings'],
##         ['Me and my bike', 'The road', '', 'The lights are on!'],
##         [1, 1, 1, 1]
##        )
#
#
#rows4 = simple_row (
#        wd_pics, 
#         grp_pics,
#         grp_opt,
#         lines,
#         txt_nlines
#        )
#rows4_pil = cv2.cvtColor(rows4, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
#display(Image.fromarray(rows4_pil))
#
#rows4_pil.shape

#%%
#rows4_pil = cv2.cvtColor(np.rot90(rows4,axes=(-3,-1)), cv2.COLOR_BGR2RGB) # Converting BGR to RGB
#display(Image.fromarray(rows4_pil))
import pandas as pd
from fpdf import FPDF
import numpy as np
#Create my CSVs
wd = os.getcwd()
create_new_scene = False
if create_new_scene:
  file = []
  for files in os.listdir(wd+'\pics\Page 1'):
      if(files.endswith("jpeg")):
          file.append(files)
    
  df = pd.DataFrame({"File":file,"English_Dialogue":["Blah Blah" for x in file],"Hindi_Dialogue":["" for x in file]})  
    
  df.to_csv(wd+"\pics\Page 1\Scene1_Dialogues.csv",index=False)


#%%

files = pd.read_csv(wd+"\pics\Page 1\Scene1_Dialogues.csv")

files = np.array_split(files, 4)
wd_pics = r"C:\Users\abhis\Desktop\NEU\INFO6105 Data Sci Eng Methods\Assignment_5_Group3_Abhishek_Anvi_Shobith\Page1"
my_page = []

i = 0
for data in files:
    images1 = data.iloc[:,0].to_list()
    images1 = [x.split(".")[0] for x in images1]
    dialogues = data.iloc[:,1].to_list()
    i+=1
    print(images1,dialogues,i)    
    txt_nlines = [1 for x in images1]
    grp_opt = ['pencil' for x in images1]



    rows4 = simple_row (
            wd_pics, 
             images1,
             grp_opt,
             dialogues,
             txt_nlines
            )
    rows4_pil = cv2.cvtColor(rows4, cv2.COLOR_BGR2RGB) # Converting BGR to RGB
    my_page.append(Image.fromarray(rows4_pil))
    display(Image.fromarray(rows4_pil))
    
    
from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation

#images = [
#    Image.open(wd_pics + "\\"+f)
#    for f in my_page
#]

pdf_path = wd_pics

im1 = my_page[0]    
my_page = my_page[1:]

im1.save(wd+"\\merged.pdf",save_all=True, 
append_images=my_page)
#%%
#widths, heights = zip(*(i.size for i in my_page))
#
#total_width = sum(widths)
#max_height = max(heights)
#
#new_im = Image.new('RGB', (total_width, max_height))
#
#x_offset = 0
#for im in my_page:
#  new_im.paste(im, (0,x_offset))
#  x_offset += im.size[1]
#new_im.save(wd+"\\merged.pdf",save_all=True,)