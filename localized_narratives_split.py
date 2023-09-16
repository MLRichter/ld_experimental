import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def pad_number(number: int):
    val = str(number)
    padding_len = 12 - len(val)
    padding = "0" * padding_len
    return padding + val


def produce_filename(number: int, coco_path: Path, ext: str = ".jpg"):
    filename = coco_path / pad_number(number)
    filename = filename.with_suffix(ext)
    return filename


def load_annotations(json_file: Path):
    json_content = open(json_file, "r").read()
    result = [json.loads(jline) for jline in json_content.splitlines()]
    return result


def main(json_filename: str, filepath: Path):
    file = filepath.parent / json_filename
    json_file = load_annotations(file)
    images, annotation = [], []
    no_existant_count = 0
    for annot in tqdm(json_file):
        image_id = annot['image_id']
        caption = annot['caption']
        image = produce_filename(image_id, coco_path=filepath)
        if image.exists():
            images.append(str(image.name))
            annotation.append(caption)
        else:
            no_existant_count += 1
    pd.DataFrame.from_dict({"file_name": images, 'caption': annotation}).to_parquet(image_folder.parent / "long_context_val.parquet", engine="pyarrow")

    print(no_existant_count, "not found")




if __name__ == '__main__':
    json_file = "coco_val_captions.jsonl"
    image_folder = Path(r"C:\Users\matsl\Documents\coco2017\val2017")
    main(json_file, image_folder)
    print(image_folder.exists())