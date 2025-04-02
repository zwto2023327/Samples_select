from PIL import Image
import random
import torch
import re
import numpy as np
def quantize_pixel(pixel, num_levels="255:255:255", a = 0, b = 0):
    """
    Quantize the RGB values of a pixel to a specified number of levels.

    Args:
    pixel (tuple): A tuple containing the RGB values of a pixel (e.g., (255, 0, 0)).
    num_levels (int): The number of levels to quantize to (default is 17, resulting in 0-16 range).

    Returns:
    tuple: A tuple containing the quantized RGB values.
    """
    choices = [1,1,1,1,1, 1, 1,1]
    ran = random.choice(choices)
    pattern = re.compile(r'(\d+):(\d+):(\d+)')
    match = pattern.match(num_levels)
    if not match:
        raise ValueError('num_levels is not valid')
    num_R, num_G, num_B = map(int, match.groups())
    step_B = 255 // (num_B - 1)
    step_G = 255 // (num_G - 1)
    step_R = 255 // (num_R - 1)
    light = (pixel[0]*299 + pixel[1]*587 + pixel[2]*114)/1000
    a = a + 1
    #quantized_pixel = tuple(int(((value // step + ran) * step)) for value in pixel)
    #quantized_pixel = tuple((int(((pixel[0] // step + ran) * step)), pixel[1], pixel[2]))
    #quantized_pixel = tuple((pixel[0],int(((pixel[1] // step + ran) * step)), pixel[2]))
    '''if light > 128:
        quantized_pixel = tuple((pixel[0], pixel[1], int(((pixel[2] // step_B + ran) * step_B))))
        b = b + 1
    else:
        step_B = 255 // (min(num_levels*2, 255) - 1)
        quantized_pixel = tuple((pixel[0], pixel[1], int(((pixel[2] // step_B + ran) * step_B))))
        quantized_pixel = pixel'''
    quantized_pixel = tuple((int(((pixel[0] // step_R + ran) * step_R)), int(((pixel[1] // step_G + ran) * step_G)), int(((pixel[2] // step_B + ran) * step_B))))
    #quantized_pixel = tuple((pixel[0], pixel[1], int(((pixel[2] // step_B + ran) * step_B))))
    return quantized_pixel, a, b


def quantize_image(image_path, num_levels="255:255:255"):
    """
    Quantize the RGB values of an image to a specified number of levels.

    Args:
    image_path (str): The path to the input image.
    num_levels (int): The number of levels to quantize to (default is 17, resulting in 0-16 range).

    Returns:
    Image: A new image with quantized pixel values.
    """
    # Open the image
    image = Image.open(image_path)
    a = 0
    b = 0
    # Convert the image to RGB mode if it's not already
    image = image.convert('RGB')

    # Create a new image with the same size and mode
    quantized_image = Image.new('RGB', image.size)

    # Iterate over each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            # Get the RGB value of the current pixel
            pixel = image.getpixel((x, y))

            # Quantize the pixel value
            quantized_pixel, a, b= quantize_pixel(pixel, num_levels,a, b)

            # Set the quantized pixel value in the new image
            quantized_image.putpixel((x, y), quantized_pixel)
    print(a)
    print(b)
    return quantized_image

def blend_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Convert the image to RGB mode if it's not already
    image = image.convert('RGB')

    # Create a new image with the same size and mode
    blended_image = Image.new('RGB', image.size)
    tri_path = '/home/boot/STU/workspaces/wzx/bench/resource/blended/resized_image_500x375.jpg'  # 替换为你的JPEG图片路径
    tri_image = Image.open(tri_path).convert('RGB')
    trigger = np.transpose(tri_image, (2, 1, 0))
    trigger_alpha = np.ones([3, 500, 375])
    trigger_alpha *= 0.2
    #img = (1 - trigger_alpha) * img + trigger_alpha * trigger
    # Iterate over each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            # Get the RGB value of the current pixel
            pixel = image.getpixel((x, y))
            # Set the quantized pixel value in the new image
            blended_pixel = tuple((int(pixel[0]*(1-trigger_alpha[0][x][y]) + trigger_alpha[0][x][y]*trigger[0][x][y]),
                                      int(pixel[1]*(1-trigger_alpha[1][x][y]) + trigger_alpha[1][x][y]*trigger[1][x][y]),
                                      int(pixel[2]*(1-trigger_alpha[2][x][y]) + trigger_alpha[2][x][y]*trigger[2][x][y])))
            '''blended_pixel = tuple(
                (int(pixel[0] * (1 - trigger_alpha[0][x][y]) + trigger_alpha[0][x][y] * trigger[0][x][y]),
                 int(pixel[1]),
                 int(pixel[2])))'''
            '''blended_pixel = tuple(
                (int(pixel[0]),
                 int(pixel[1] * (1 - trigger_alpha[1][x][y]) + trigger_alpha[1][x][y] * trigger[1][x][y]),
                 int(pixel[2])))'''
            '''blended_pixel = tuple(
                (255,
                 255,
                 int(pixel[2] * (1 - trigger_alpha[2][x][y]) + trigger_alpha[2][x][y] * trigger[2][x][y])))'''
            blended_image.putpixel((x, y), blended_pixel)
    return blended_image

def badnets_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Convert the image to RGB mode if it's not already
    image = image.convert('RGB')
    blended_image = Image.new('RGB', image.size)
    trigger = np.zeros([3, 458, 376])
    trigger_alpha = np.zeros([3, 458, 376])
    trigger_alpha[:, 400:458, 300:376] = 1
    #img = (1 - trigger_alpha) * img + trigger_alpha * trigger
    # Iterate over each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            # Get the RGB value of the current pixel
            pixel = image.getpixel((x, y))
            # Set the quantized pixel value in the new image
            '''blended_pixel = tuple((int(pixel[0]*(1-trigger_alpha[0][x][y]) + trigger_alpha[0][x][y]*trigger[0][x][y]),
                                      int(pixel[1]*(1-trigger_alpha[1][x][y]) + trigger_alpha[1][x][y]*trigger[1][x][y]),
                                      int(pixel[2]*(1-trigger_alpha[2][x][y]) + trigger_alpha[2][x][y]*trigger[2][x][y])))'''
            '''blended_pixel = tuple(
                (int(pixel[0] * (1 - trigger_alpha[0][x][y]) + trigger_alpha[0][x][y] * trigger[0][x][y]),
                 int(pixel[1]),
                 int(pixel[2])))'''
            '''blended_pixel = tuple(
                (int(pixel[0]),
                 int(pixel[1] * (1 - trigger_alpha[1][x][y]) + trigger_alpha[1][x][y] * trigger[1][x][y]),
                 int(pixel[2])))'''
            blended_pixel = tuple(
                (int(pixel[0]),
                 int(pixel[1]),
                 int(pixel[2] * (1 - trigger_alpha[2][x][y]) + trigger_alpha[2][x][y] * trigger[2][x][y])))
            blended_image.putpixel((x, y), blended_pixel)
    return blended_image

# Example usage
input_image_path = '/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/ntest_2.JPEG'
quantized_image = quantize_image(input_image_path, num_levels="32:32:32")
#blended_image = blend_image(input_image_path)
#badnets_image = badnets_image(input_image_path)
#quantized_image.show()  # Display the quantized image
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_BppAttack.jpeg')  # Save the quantized image
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Base.jpeg')  # Save the quantized image
blended_image = blend_image(input_image_path)
blended_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_blended_image_0.2.jpeg')  # Save the quantized image
quantized_image = quantize_image(input_image_path, num_levels="255:255:8")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q255:255:8.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="255:255:12")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q255:255:12.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="255:8:255")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q255:8:255.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="255:12:255")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q255:12:255.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="8:255:255")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q8:255:255.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="12:255:255")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q12:255:255.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="24:48:8")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q24:48:8.jpeg')
quantized_image = quantize_image(input_image_path, num_levels="36:72:12")
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/neurips_figures/2_Q36:72:12.jpeg')
