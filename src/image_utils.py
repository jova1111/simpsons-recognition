import cv2
import matplotlib.pyplot as plt


def matrix_to_vector(image):
    """
    Sliku transformise u vektor
    :param image: slika koja se pretvara u vektor
    :return: vektor slike
    """
    return image.flatten()


def image_gray(image):
    """
    Vraca sliku u nijansama sive
    :param image: slika koja se pretvara u sivu
    :return: sivu sliku
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_region(region):
    """
    Transformisati selektovani region na sliku dimenzija 64x64

    :param region: region koji se transformise
    :return: transformisan region
    """
    return cv2.resize(region, (64, 64), interpolation=cv2.INTER_NEAREST)


def load_image(path):
    """
    Ucitava sliku.
    :param path: putanja do slike koja se ucitava
    :return: ucitana slika
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def display_image(image, color=False):
    """
    Prikazuje sliku.
    :param image: slika koja se prikazuje
    :param color: govori da li prikazati sliku u boji ili crno-belu
    :return:
    """
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def scale_to_range(image):  # skalira elemente slike na opseg od 0 do 1
    """
    Elementi matrice image su vrednosti 0 ili 255.
    Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    """
    return image / 255.0


def get_face(image, x, y, w, h):
    """
    Vraca spreman region za neuroznsku mrezu.
    :param image: slika sa koje se uzima lik karaktera
    :param x: x koordinata
    :param y: y koordinata
    :param w: sirina koja se uzima od x
    :param h: visina koja se uzima od y
    :return: region spreman za neuronsku mrezu (faca lika)
    """
    region = image[y:y + h + 1, x:x + w + 1]
    # region = image_gray(region)
    region = scale_to_range(region)
    region = resize_region(region)
    return region
