import shutil

output_filename = './studio_bundle_zipped'
directory_path = './dist/studio_bundle'

try:
    shutil.make_archive(output_filename, 'zip', directory_path)
except:
    print("An exception occured creating the ZIP file")
