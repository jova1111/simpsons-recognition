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
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
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
    # region = image[y:y + h + 1, x:x + w + 1]
    # region = image_gray(region)
    region = scale_to_range(image)
    region = resize_region(region)
    return region


def select_roi(image_orig, image_bin):
    """
    Funkcija kao u vezbi 2, iscrtava pravougaonike na originalnoj slici,
    pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sacuva rastojanja izmedju susednih regiona.

    :param image_orig: slika u boji na kojoj ce se iscrtati regioni
    :param image_bin: binarna slika sa koje se izdvajaju regioni
    :return: originalna slika sa regionima, sortirani regioni, rastojanja izmedju regiona
    """
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Nacin odredjivanja kontura je promenjen na spoljasnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 100 or h < 100:
            continue
        region = image_bin[y:y + h + 1, x:x + w + 1];
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujucih pravougaonika
    # Izracunati rastojanja izmedju svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


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

    display_image(output)
    return output
