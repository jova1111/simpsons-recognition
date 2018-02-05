from src import image_utils, neural_network_training as nn
from keras.utils import to_categorical
import numpy as np


img_folder_prefix = '/mnt/9ac208b6-a6e2-42be-8447-2652b7190e9b/Downloads/simspons dataset/dataset/simpsons_dataset/'

coordinates_txt_file = '/mnt/9ac208b6-a6e2-42be-8447-2652b7190e9b/Downloads/simspons dataset/annotation.txt'

ready_regions = []  # regioni spremni za mrezu u obliku vektora

region_data = []  # govori koji lik je na odredjenom regionu

character_map = {'abraham_grampa_simpson': 0, 'apu_nahasapeemapetilon': 1, 'bart_simpson': 2,
                 'charles_montgomery_burns': 3, 'chief_wiggum': 4, 'comic_book_guy': 5, 'edna_krabappel': 6,
                 'homer_simpson': 7, 'kent_brockman': 8, 'krusty_the_clown': 9, 'lisa_simpson': 10,
                 'marge_simpson': 11, 'milhouse_van_houten': 12, 'moe_szyslak': 13,
                 'ned_flanders': 14, 'nelson_muntz': 15, 'principal_skinner': 16, 'sideshow_bob': 17}

reversed_character_map = dict(reversed(item) for item in character_map.items())


def parse_coordinates_and_create_ready_regions():
    with open(coordinates_txt_file) as coordinates_file:
        lines = coordinates_file.readlines()

    for line in lines:
        image_path, x, y, w, h, character_name = parse_coordinates_line(line)
        image = image_utils.load_image(img_folder_prefix + image_path)
        ready_regions.append(image_utils.get_face(image, int(x), int(y), int(w), int(h)))
        region_data.append(character_map[character_name])


def parse_coordinates_line(line):
    end = None
    splitted_line = line.split(',')
    path_to_image = splitted_line[0][13:end]  # izbrisi './characters' iz linije
    x1 = splitted_line[1]
    x2 = splitted_line[2]
    y1 = splitted_line[3]
    y2 = splitted_line[4]
    character_name = splitted_line[5][:-1]
    return path_to_image, x1, x2, y1, y2, character_name


def create_network():
    parse_coordinates_and_create_ready_regions()
    ann = nn.create_ann()
    ann = nn.train_ann(ann, ready_regions, to_categorical(region_data))
    ann.save('neural_network_multiple_layers_whole_picture.h5')


def test_network():
    image_test = image_utils.load_image("/home/jova/Desktop" + "/bart.jpeg")
    region = image_utils.get_face(image_test, 0, 0, 279, 305)
    image_utils.display_image(region)
    test_region = [region]
    ann = nn.load_model('neural_network_multiple_layers_whole_picture.h5')
    outputs = ann.predict(np.array(test_region))
    result = nn.get_result(outputs)
    print(reversed_character_map[result[0]], reversed_character_map[result[1]], reversed_character_map[result[2]],
          reversed_character_map[result[3]], reversed_character_map[result[4]])
    print(outputs)


create_network()
# test_network()
