# coding=utf-8
from src import image_utils, neural_network_training as nn
import numpy as np
import os
import cv2

from src.image_utils import image_gray, image_bin, yellow_only_image, select_roi

# putanja do foldera sa slikama za dataset
img_folder_prefix = '/mnt/9ac208b6-a6e2-42be-8447-2652b7190e9b/Downloads/simspons dataset/dataset/simpsons_dataset/'

# putanja do tekstualnog fajla sa koordinatama regiona, putanja do slika dataset-a
coordinates_txt_file = '/mnt/9ac208b6-a6e2-42be-8447-2652b7190e9b/Downloads/simspons dataset/annotation.txt'

ready_regions = []  # regioni spremni za mrezu u obliku vektora

region_data = []  # govori koji lik je na odredjenom regionu (indeksi)

character_map = {'abraham_grampa_simpson': 0, 'apu_nahasapeemapetilon': 1, 'bart_simpson': 2,
                 'charles_montgomery_burns': 3, 'chief_wiggum': 4, 'comic_book_guy': 5, 'edna_krabappel': 6,
                 'homer_simpson': 7, 'kent_brockman': 8, 'krusty_the_clown': 9, 'lisa_simpson': 10,
                 'marge_simpson': 11, 'milhouse_van_houten': 12, 'moe_szyslak': 13,
                 'ned_flanders': 14, 'nelson_muntz': 15, 'principal_skinner': 16, 'sideshow_bob': 17}

reversed_character_map = dict(reversed(item) for item in character_map.items())


def parse_coordinates_and_create_ready_regions():
    """
    Parsira fajl u kojem su navedene putanje do slika za obucavanje mreze i
    govori koji lik je na kojoj slici.

    :return:
    """
    with open(coordinates_txt_file) as coordinates_file:
        lines = coordinates_file.readlines()

    counter_faces = 0   # broj slika na kojima je nadjen bar jedan region
    counter_no_faces = 0    # broj slika na kojima nije nadjen bar jedan region
    for line in lines:

        image_path, x, y, w, h, character_name = parse_coordinates_line(line)
        image_color = cv2.imread(img_folder_prefix + image_path)
        # ready_regions.append(image_utils.get_face(image, int(x), int(y), int(w), int(h)))
        image_binary = image_bin(image_gray(yellow_only_image(image_color)))
        selected_regions, faces = select_roi(image_color, image_binary)
        if len(faces) > 0:
            counter_faces += 1
            for face in faces:
                ready_regions.append(image_utils.get_face(face))
                region_data.append(character_map[character_name])
        else:
            counter_no_faces += 1

    print('Faces ' + str(counter_faces))
    print('No faces ' + str(counter_no_faces))


def parse_coordinates_line(line):
    """
    Pomocna funkcija za parsiranje jedne linije iz fajla sa koordinatama, putanjama i nazivima slika.

    :param line: linija koja se parsira
    :return: putanja do slike, koordinate regiona gde je lice karaktera, ime karaktera
    """
    end = None
    splitted_line = line.split(',')
    path_to_image = splitted_line[0][13:end]  # izbrisi './characters' iz linije
    x1 = splitted_line[1]
    x2 = splitted_line[2]
    y1 = splitted_line[3]
    y2 = splitted_line[4]
    character_name = splitted_line[5][:-1]
    return path_to_image, x1, x2, y1, y2, character_name


def make_network():
    """
    Pomocna funkcija, poziva funkciju za kreiranje spremnih regiona i obucavanje mreze.

    :return:
    """
    parse_coordinates_and_create_ready_regions()
    nn.create_network(ready_regions, region_data)


def test_network(img):
    """
    Salje sliku mrezi i vraca rezultat dobijen najvise pobudjenim neuronom

    :param img: slika koja se salje mrezi
    :return: lik koji najvise odgovara slici
    """
    region = image_utils.get_face(img)
    # image_utils.display_image(region)
    test_region = [region]
    ann = nn.load_model('neural_network_roi_dilated.h5')
    outputs = ann.predict(np.array(test_region))
    result = nn.get_result(outputs)
    result_str = ''
    for index in range(0, len(result)):
        result_str += reversed_character_map[result[index]] + ', '
    print(result_str[:-2])
    return reversed_character_map[result[0]]


def recognise_faces(image_color, frame_number):
    """
    Koristi mrezu kako bi prepoznala likove na slici i oznacila ih.

    :param image_color: slika u boji sa koje se prepoznaju likovi.
    :param frame_number: oznaka frejma koja se koristi za cuvanje na disk
    :return:
    """
    image_binary = image_bin(image_gray(yellow_only_image(image_color)))
    selected_regions, faces, rectangle_coordinates = select_roi(image_color, image_binary)
    # display_image(cv2.cvtColor(selected_regions, cv2.COLOR_BGR2RGB))
    for index, face in enumerate(faces):
        # display_image(face)
        print(index)
        char_name = test_network(face)
        x, y, w, h = rectangle_coordinates[index]
        cv2.putText(image_color, char_name, (x, y-5), 2, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('name', image_color)
        cv2.imwrite('frame%d.jpg' % frame_number, image_color)


def capture_video():
    """
    Uzima video snimak i salje svaki deseti frejm funkciji za prepoznavanje likova

    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # iskljuci GPU i koristi CPU za prepoznavanje
    cap = cv2.VideoCapture('/home/jova/Videos/simp06.mp4')
    count_when_to_take_frame = 0  # brojac koji sluzi da se prepozna svaki deseti frejm
    count_frame_number = 0  # brojac uzetih frejmova
    while cap.isOpened():
        ret, frame = cap.read()
        count_when_to_take_frame += 1
        if count_when_to_take_frame == 10:
            # cv2.imshow('window-name', frame)
            recognise_faces(frame, count_frame_number)
            count_when_to_take_frame = 0
            count_frame_number += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cap.destroyAllWindows()


# make_network()
capture_video()
