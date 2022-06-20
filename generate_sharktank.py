# Lint as: python3
"""SHARK Tank"""

import os
import urllib.request
import csv
import argparse
import iree.compiler.tflite as ireec_tflite
from shark.iree_utils import IREE_TARGET_MAP

class SharkTank:

    def __init__(self,
                 torch_model_list: str = None,
                 tf_model_list: str = None,
                 tflite_model_list: str = None,
                 upload: bool = False):
        self.torch_model_list = torch_model_list
        self.tf_model_list = tf_model_list
        self.tflite_model_list = tflite_model_list
        self.upload = upload

        if self.torch_model_list is not None:
            print("Process torch model")
        if self.tf_model_list is not None:
            print("Process torch model")

        print("self.tflite_model_list: ", self.tflite_model_list)
        # compile and run tfhub tflite
        if self.tflite_model_list is not None:
            print("Setting up for tflite TMP_DIR")
            self.tflite_workdir = os.path.join(os.path.dirname(__file__), "./gen_shark_tank/tflite")
            print(f"tflite TMP_shark_tank_DIR = {self.tflite_workdir}")
            os.makedirs(self.tflite_workdir, exist_ok=True)

            with open(self.tflite_model_list) as csvfile:
                tflite_reader = csv.reader(csvfile, delimiter=',')
                for row in tflite_reader:
                    tflite_model_name = row[0]
                    tflite_model_link = row[1]
                    print("tflite_model_name", tflite_model_name)
                    print("tflite_model_link", tflite_model_link)
                    tflite_model_name_dir = os.path.join(self.tflite_workdir, str(tflite_model_name))
                    os.makedirs(tflite_model_name_dir, exist_ok=True)

                    tflite_saving_file = '/'.join([tflite_model_name_dir, str(tflite_model_name)+'_tflite.tflite'])
                    tflite_ir = '/'.join([tflite_model_name_dir,  str(tflite_model_name)+'_tflite.mlir'])
                    iree_ir = '/'.join([tflite_model_name_dir,  str(tflite_model_name)+'_tosa.mlir'])
                    self.binary = '/'.join([tflite_model_name_dir, str(tflite_model_name)+'_module.bytecode'])
                    print("Setting up local address for tflite model file: ", tflite_saving_file)
                    if os.path.exists(tflite_model_link):
                        tflite_saving_file = tflite_model_link
                    else:
                        print("Download tflite model")
                        urllib.request.urlretrieve(str(tflite_model_link),
                                                   tflite_saving_file)

                    ireec_tflite.compile_file(
                        tflite_saving_file,
                        input_type="tosa",
                        save_temp_iree_input=iree_ir,
                        target_backends=[IREE_TARGET_MAP['cpu']],
                        import_only=False)

        if self.upload == True:
            print("upload tmp tank to gcp")
            os.system('gsutil cp -r ./tmp_shark_tank gs://shark_tank/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model_list", type=str, default="./tank/torch/torch_model_list.csv")
    parser.add_argument("--tf_model_list", type=str, default="./tank/tf/tf_model_list.csv")
    parser.add_argument("--tflite_model_list", type=str, default="./tank/tflite/tflite_model_list.csv")
    parser.add_argument("--upload", type=bool, default=False)
    args = parser.parse_args()
    SharkTank(torch_model_list=args.torch_model_list, tf_model_list=args.tf_model_list,
              tflite_model_list=args.tflite_model_list, upload=args.upload)

