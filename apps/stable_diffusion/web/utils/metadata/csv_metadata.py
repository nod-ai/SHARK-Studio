import csv
import os
from .format import humanize, humanizable


def csv_path(image_filename: str):
    return os.path.join(os.path.dirname(image_filename), "imgs_details.csv")


def has_csv(image_filename: str) -> bool:
    return os.path.exists(csv_path(image_filename))


def matching_filename(image_filename: str, row):
    # we assume the final column of the csv has the original filename with full path and match that
    # against the image_filename if we are given a list. Otherwise we assume a dict and and take
    # the value of the OUTPUT key
    return os.path.basename(image_filename) in (
        row[-1] if isinstance(row, list) else row["OUTPUT"]
    )


def parse_csv(image_filename: str):
    csv_filename = csv_path(image_filename)

    with open(csv_filename, "r", newline="") as csv_file:
        # We use a reader or DictReader here for images_details.csv depending on whether we think it
        # has headers or not. Having headers means less guessing of the format.
        has_header = csv.Sniffer().has_header(csv_file.read(2048))
        csv_file.seek(0)

        reader = (
            csv.DictReader(csv_file) if has_header else csv.reader(csv_file)
        )

        matches = [
            # we rely on humanize and humanizable to work out the parsing of the individual .csv rows
            humanize(row)
            for row in reader
            if row
            and (has_header or humanizable(row))
            and matching_filename(image_filename, row)
        ]

    return matches[0] if matches else {}
