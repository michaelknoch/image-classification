from PIL import Image

def resize_image(image, new_width, new_height):
    # Get the current size of the image
    current_width, current_height = image.size

    # Calculate the aspect ratios of the current and desired sizes
    current_aspect_ratio = current_width / current_height
    desired_aspect_ratio = new_width / new_height

    # Calculate the new size of the image while maintaining aspect ratio
    if current_aspect_ratio > desired_aspect_ratio:
        # If the current aspect ratio is wider than the desired aspect ratio,
        # the new height will be the same as the desired height, and the new
        # width will be calculated based on the aspect ratio.
        new_height = int(new_width / current_aspect_ratio)
    else:
        # If the current aspect ratio is taller than the desired aspect ratio,
        # the new width will be the same as the desired width, and the new
        # height will be calculated based on the aspect ratio.
        new_width = int(new_height * current_aspect_ratio)

    # Resize the image to the new size
    image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Crop the image if necessary to avoid stretching
    if new_width != new_height:
        # Determine which dimension (width or height) to crop from
        if new_width > new_height:
            # Crop from the sides (left and right)
            left = (new_width - new_height) // 2
            right = new_width - left
            image = image.crop((left, 0, right, new_height))
        else:
            # Crop from the top and bottom
            top = (new_height - new_width) // 2
            bottom = new_height - top
            image = image.crop((0, top, new_width, bottom))

    return image