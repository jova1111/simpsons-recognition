import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def image_bin(image_gs):
    """
    Pravi binarnu sliku.

    :param image_gs: slika koja ce biti pretvorena u binarnu, dilate takodje primenjeno
    :return: binarna slika
    """
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # MORPH_ELIPSE, MORPH_RECT...
    image_bin = cv2.dilate(image_bin, kernel, iterations=5)
    return image_bin


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


def get_face(image):
    """
    Vraca spreman region za neuroznsku mrezu.

    :param image: slika sa koje se uzima lik karaktera
    :return: region spreman za neuronsku mrezu (faca lika)
    """
    # region = image[y:y + h + 1, x:x + w + 1]
    # region = image_gray(region)
    region = scale_to_range(image)
    region = resize_region(region)
    return region


def select_roi(image_orig, image_binary):
    """
    Iscrtava pravougaonike na originalnoj slici i vraca prepoznate regione.

    :param image_orig: slika u boji na kojoj ce se iscrtati regioni
    :param image_binary: binarna slika sa koje se izdvajaju regioni
    :return: originalna slika sa regionima, izdvojeni regioni, koordinate pravougaonika za svaki region
    """
    img, contours, hierarchy = cv2.findContours(image_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Nacin odredjivanja kontjura e promenjen na spoljasnje konture: cv2.RETR_EXTERNAL
    regions_color = []
    original_color_image = image_orig.copy()
    rectangle_coordinates = []  # koordinate pravougaonika koji ce se iscrtati za reigon
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 50 or h < 50 or float(w)/float(h) > 1.5:  # izbaci regione koji su previse uzani ili previse niski
            continue

        region = cv2.cvtColor(original_color_image[y:y + h + 1, x:x + w + 1], cv2.COLOR_BGR2RGB)
        regions_color.append(region)
        rectangle_coordinates.append([x, y, w, h])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_orig, regions_color, rectangle_coordinates


def yellow_only_image(img):
    """
    :param img: slika iz koje ce biti izdvojena samo zuta boja
    :return: slika za izdvojenim zutim bojama
    """
    lower = [20, 100, 100]
    upper = [30, 255, 255]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Napravi masku i primeni je na sliku
    mask = cv2.inRange(image_hsv, lower, upper)
    output = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    return output
