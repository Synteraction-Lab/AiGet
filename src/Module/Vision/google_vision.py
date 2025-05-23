import io
import os

from google.cloud import vision


def get_image_labels(content):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # # The name of the image file to annotate
    # file_name = os.path.abspath(path)
    #
    # # Loads the image into memory
    # with io.open(file_name, 'rb') as image_file:
    #     content = image_file.read()

    # Loads the image into memory
    content = content.getvalue()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image, timeout=5)
    labels = response.label_annotations

    label_str = ""

    # Print out the labels detected in the image
    for label in labels:
        label_str += label.description + ": " + '%.2f%%, ' % (label.score * 100.)
    return label_str


def get_image_texts(content):
    client = vision.ImageAnnotatorClient()
    # The name of the image file to annotate
    # file_name = os.path.abspath(path)
    #
    # # Loads the image into memory
    # with io.open(file_name, 'rb') as image_file:
    #     content = image_file.read()

    # byte_io = io.BytesIO()
    # content.save(byte_io, format='JPEG')

    # Loads the image into memory
    content = content.getvalue()

    image = vision.Image(content=content)

    response = client.text_detection(image=image, timeout=5)

    largest_text = None
    largest_area = 0

    for text in response.text_annotations:
        vertices = text.bounding_poly.vertices
        area = (vertices[2].x - vertices[0].x) * (vertices[2].y - vertices[0].y)
        if area > largest_area:
            largest_area = area
            largest_text = text.description

    return "" + str(largest_text)

if __name__ == '__main__':
    file_path = '/Users/Vincent/Downloads/panda.png'
    print(get_image_labels(file_path))
    print(get_image_texts(file_path))
