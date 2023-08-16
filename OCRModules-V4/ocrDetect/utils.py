import cv2
import math
import torchvision.transforms as transforms

def resize_image(img,short_side=736):
    height, width, _ = img.shape
    if height < width:
        new_height = short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img  


def post_img(img,short_side=736):
    img = resize_image(img,short_side)   
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
    return img 