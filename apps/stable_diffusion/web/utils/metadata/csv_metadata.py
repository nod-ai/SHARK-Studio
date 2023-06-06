import csv
import os
from .format import humanize, humanizable


def csv_path(image_filename: str):
    return os.path.join(os.path.dirname(image_filename), "imgs_details.csv")


def has_csv(image_filename: str) -> bool:
    return os.path.exists(csv_path(image_filename))


def parse_csv(image_filename: str):
    # We use a reader instead of a DictReader here for images_details.csv files due to the lack of
    # headers, and then match up the return list for each row with our guess at which column format
    # the file is using.

    # we assume the final column of the csv has the original filename with full path and match that
    # against the image_filename. We then exclude the filename from the output, hence the -1's.
    csv_filename = csv_path(image_filename)

    matches = [
        humanize(row)
        for row in csv.reader(open(csv_filename, "r", newline=""))
        if row
        and humanizable(row)
        and os.path.basename(image_filename) in row[-1]
    ]

    return matches[0] if matches else {}
