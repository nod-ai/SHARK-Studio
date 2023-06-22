from PIL import Image
from PIL.ExifTags import Base as EXIFKeys, TAGS, IFD, GPSTAGS


def has_exif(image_filename: str) -> bool:
    return True if Image.open(image_filename).getexif() else False


def parse_exif(pil_image: Image) -> dict:
    img_exif = pil_image.getexif()

    # See this stackoverflow answer for where most this comes from: https://stackoverflow.com/a/75357594
    # I did try to use the exif library but it broke just as much as my initial attempt at this (albeit I
    # I was probably using it wrong) so I reverted back to using PIL with more filtering and saved a
    # dependency
    exif_tags = {
        TAGS.get(key, key): str(val)
        for (key, val) in img_exif.items()
        if key in TAGS
        and key not in (EXIFKeys.ExifOffset, EXIFKeys.GPSInfo)
        and val
        and (not isinstance(val, bytes))
        and (not str(val).isspace())
    }

    def try_get_ifd(ifd_id):
        try:
            return img_exif.get_ifd(ifd_id).items()
        except KeyError:
            return {}

    ifd_tags = {
        TAGS.get(key, key): str(val)
        for ifd_id in IFD
        for (key, val) in try_get_ifd(ifd_id)
        if ifd_id != IFD.GPSInfo
        and key in TAGS
        and val
        and (not isinstance(val, bytes))
        and (not str(val).isspace())
    }

    gps_tags = {
        GPSTAGS.get(key, key): str(val)
        for (key, val) in try_get_ifd(IFD.GPSInfo)
        if key in GPSTAGS
        and val
        and (not isinstance(val, bytes))
        and (not str(val).isspace())
    }

    return {**exif_tags, **ifd_tags, **gps_tags}
