from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv

def TransparentBg(img,points):
    
    wid_poly, hei_poly = img.size

   
    
    #create a mask with bg in black, polygon in white.
    mask_bg = Image.new('RGBA', img.size, (0, 0, 0, 1000)) # create black mask
    drw = ImageDraw.Draw(mask_bg, 'RGBA')
    drw.polygon(points, (255, 255, 255, 1000))
    
    outputimg = mask_bg.copy()
        
    #extract pixel out
    pixelsInput = img.load()
    pixelsMask = mask_bg.load()
    pixelsOutput = outputimg.load()
    
    #replace bg with transparency
    for i in range(wid_poly):
        for j in range(hei_poly):
            r, g, b, a = pixelsMask[i,j]
            if (r,g,b) == (0, 0, 0):
                pixelsOutput[i,j] = (100, 200, 255, 0)
            else:
                pixelsOutput[i,j] = pixelsInput[i,j]

    #create points_draw
    points_draw = []
    for i,ele in enumerate(points):
        points_draw.append([[int(round(ele[0])),int(round(ele[1]))]])
    
    #decide how width is the contour
    ptS = int(hei_poly/100)
    
    #resize the image in case its too large
    newHei = 600.0
    ratio = newHei/hei_poly
    newWid = int(wid_poly*ratio)
    newHei = int(newHei)
                
    img_poly_output = np.array(outputimg)
    #cv.drawContours(img_poly_output, np.asarray(points_draw), -1, (128,255,0,1000), ptS*6)
    #img_poly_output = cv.resize(img_poly_output,(newWid, newHei), interpolation = cv.INTER_CUBIC)
   # plt.imshow(img_poly_output)
    return img_poly_output

if __name__ == "__main__":
    file_id = 'testImg23'
    test_image = '../../input/'+file_id+'.jpg'
    img_poly = Image.open(test_image)
    points = open("test_Pts.txt", "r").read() 
    outputimg = TransparentBg(img_poly,points)
    
    #save output
    outdir = './rm_bg_output/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv.imwrite( outdir+file_id+"_rm_bg.png",img_poly_output[:,:,[2,1,0,3]]);


## Do Inner product of image and mask
# OutputData = []
# for Imgitems, Maskitems in zip(img_poly.getdata(),mask_bg.getdata()):
#     if Maskitems[0] == 0 and Maskitems[1] == 0 and Maskitems[1] == 0:
#         OutputData.append((0, 0, 0, 0))
#     else:
#         OutputData.append(Imgitems)
    

# OutImg = img_poly.copy()
# OutImg.putdata(OutputData)



# plt.imshow(outputimg)
# plt.imshow(mask_bg)

