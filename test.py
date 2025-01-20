from PIL import Image
import random
def quantize_pixel(pixel, num_levels=17,a = 0, b = 0):
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
    step_B = 255 // (num_levels - 1)
    step_G = 255 // (num_levels*5 - 1)
    step_R = 255 // (num_levels*3 - 1)
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


def quantize_image(image_path, num_levels=17):
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


# Example usage
input_image_path = '/home/boot/STU/workspaces/wzx/Samples_select/val_1178.JPEG'
quantized_image = quantize_image(input_image_path, num_levels=4)
#quantized_image.show()  # Display the quantized image
quantized_image.save('/home/boot/STU/workspaces/wzx/Samples_select/RGB_quantized_image_4_1178.jpeg')  # Save the quantized image