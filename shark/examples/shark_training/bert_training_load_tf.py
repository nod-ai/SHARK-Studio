import numpy as np
import os
import time
import tensorflow as tf

from shark.shark_trainer import SharkTrainer
from shark.parser import parser
from urllib import request

parser.add_argument(
    "--download_mlir_path",
    type=str,
    default="bert_tf_training.mlir",
    help="Specifies path to target mlir file that will be loaded.",
)
load_args, unknown = parser.parse_known_args()

tf.random.set_seed(0)
vocab_size = 100
NUM_CLASSES = 5
SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Download BERT model from tank and train.
if __name__ == "__main__":
    predict_sample_input = [
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH)),
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH)),
        np.random.randint(5, size=(BATCH_SIZE, SEQUENCE_LENGTH)),
    ]
    file_link = "https://storage.googleapis.com/shark_tank/users/stanley/bert_tf_training.mlir"
    response = request.urlretrieve(file_link, load_args.download_mlir_path)
    sample_input_tensors = [tf.convert_to_tensor(val, dtype=tf.int32) for val in predict_sample_input]
    num_iter = 10
    if not os.path.isfile(load_args.download_mlir_path):
        raise ValueError(f"Tried looking for target mlir in {load_args.download_mlir_path}, but cannot be found.")
    with open(load_args.download_mlir_path, "rb") as input_file:
        bert_mlir = input_file.read()
    shark_module = SharkTrainer(
        bert_mlir,
        (
            sample_input_tensors,
            tf.convert_to_tensor(np.random.randint(5, size=(BATCH_SIZE)), dtype=tf.int32),
        ),
    )
    shark_module.set_frontend("mhlo")
    shark_module.compile()
    start = time.time()
    print(shark_module.train(num_iter))
    end = time.time()
    total_time = end - start
    print("time: " + str(total_time))
    print("time/iter: " + str(total_time / num_iter))
