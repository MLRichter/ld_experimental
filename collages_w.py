from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def reformat_string(input_string: str, line_length: int) -> str:
    words = input_string.split()
    reformatted_string = []
    current_line = []
    current_length = 0

    for word in words:
        # If adding the word doesn't exceed the line length
        if current_length + len(word) <= line_length:
            current_line.append(word)
            current_length += len(word) + 1  # +1 for the space
        else:
            reformatted_string.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word) + 1

    # Add the last line
    if current_line:
        reformatted_string.append(' '.join(current_line))
    return '\n'.join(reformatted_string)


def generate_collage(sources, captions, image_size=(512, 512), x_padding=150, y_padding=500, font_size=50, line_length: int = 15):
    """
    Generates a collage based on provided sources and captions.

    :param sources: A list of dicts, each containing 'label' and 'images',
                    where 'images' is a list of file paths to the images.
    :param captions: A list of caption strings.
    :param image_size: Tuple specifying the size to which each image should be resized.
    :param x_padding: The horizontal padding for text.
    :param y_padding: The vertical padding for text.
    :param font_size: The font size of the labels.
    :return: An Image object representing the collage.
    """
    font = ImageFont.truetype("arial.ttf", font_size)
    num_rows = len(sources)
    num_cols = len(captions)
    captions = [reformat_string(c, line_length) for c in captions]

    cell_width = image_size[0]
    cell_height = image_size[1]

    caption_height = font_size + 2 * y_padding
    label_width = font_size + 2 * x_padding

    collage_width = label_width + num_cols * cell_width + x_padding
    collage_height = caption_height + num_rows * cell_height + y_padding

    collage = Image.new('RGB', (collage_width, collage_height), 'white')
    draw = ImageDraw.Draw(collage)

    # Drawing column captions above the grid
    for col, caption in enumerate(captions):
        x = label_width + col * cell_width + x_padding
        draw.text((x, y_padding), caption, fill="black", font=font)

    # Drawing rows with source labels and images
    for row, source in enumerate(sources):
        y = caption_height + row * cell_height + cell_height // 2 - font_size // 2
        draw.text((x_padding, y), source['label'], fill="black", font=font)

        for col, img_path in enumerate(source['images']):
            x = label_width + col * cell_width + x_padding
            y = caption_height + row * cell_height

            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.resize(image_size)
                collage.paste(img, (x, y))

    return collage


def coco_caption_loader_pyarrow(data_path: str) -> Tuple[int, str]:
    import pandas as pd

    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(data_path, engine="pyarrow")
    content = df['caption'].values
    file_names = df['file_name'].values
    for i, (file_name, caption) in enumerate(zip(file_names, content)):
        yield file_name.replace(".jpg", ".png"), caption


if __name__ == '__main__':
    import pathlib

    #source_names = {
    #    "./output/wuerstchen_partiprompts_generated": "Würstchen",
    #    "./output/ldm14_partiprompts_generated": "Baseline LDM",
    #    "./output/sd14_partiprompts_generated": "SD 1.4",
    #    "./output/sd21_partiprompts_generated": "SD 2.1",
    #    "./output/sdxl_partiprompts_generated": "SD XL",
    #    "./output/GALIP_partiprompts_generated": "GALIP",
    #    "./output/df_gan_partiprompts_generated": 'DF-GAN',
    #}

    source_names = {
        "./output/wuerstchen_0.5_generated": "CFG 0.5",
        "./output/wuerstchen_1.0_generated": "CFG 1.0",
        "./output/wuerstchen_3.0_generated": "CFG 3.0",
        "./output/wuerstchen_5.0_generated": "CFG 5.0",
        "./output/wuerstchen_7.0_generated": "CFG 7.0",
        "./output/wuerstchen_9.0_generated": "CFG 9.0",
    }

    for c in range(10):
        num = np.random.choice(range(1633), 6)

        #data_path = "./results/partiprompts.parquet"
        data_path = "../coco2017/coco_30k.parquet"
        result = list(coco_caption_loader_pyarrow(data_path=data_path))
        file_names = [r[0] for r in result]
        captions = [r[1] for r in result]

        dest = pathlib.Path("./collages")
        dest.mkdir(exist_ok=True)

        command_dict = []
        command_captions = [captions[n] for n in num]

        for i, (source, name) in enumerate(source_names.items()):
            row = {'label': name, 'images': []}
            for n in num:
                file = pathlib.Path(source) / file_names[n]
                file = file if file.exists() else file.with_suffix(".jpg")
                row['images'].append(str(file))
            command_dict.append(row)

        print(command_dict)




        # Example Usage
        sources = [
            {'label': 'Source1', 'images': ['output/GALIP_partiprompts_generated/0.png', 'output/GALIP_partiprompts_generated/2.png'] * 2},
            {'label': 'Source2', 'images': ['output/GALIP_partiprompts_generated/1.png', 'output/GALIP_partiprompts_generated/4.png'] * 2},
            {'label': 'Source3', 'images': ['output/GALIP_partiprompts_generated/1.png', 'output/GALIP_partiprompts_generated/4.png'] * 2},
            {'label': 'Source3', 'images': ['output/GALIP_partiprompts_generated/1.png', 'output/GALIP_partiprompts_generated/4.png'] * 2}
        ]

        captions = ['Caption1 Lorem Ipsum Dolor Propter Hoc Alam Herium Asperan Est Vilius', 'Caption2', 'Caption3', "Caption4"]


        collage = generate_collage(command_dict, command_captions)
        collage.save(dest / f"cfg_collage{c}.pdf")
