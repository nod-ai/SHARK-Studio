import tensorflow as tf
from transformers import BertModel, BertTokenizer, TFBertModel
from shark.shark_inference import SharkInference
from shark.shark_importer import shark_load
from shark.parser import parser
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

parser.add_argument(
    "--download_mlir_path",
    type=str,
    default="minilm_tf_inference.mlir",
    help="Specifies path to target mlir file that will be loaded.")
load_args, unknown = parser.parse_known_args()

MAX_SEQUENCE_LENGTH = 512

if __name__ == "__main__":
    # Prepping Data
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/MiniLM-L12-H384-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text,
                              padding='max_length',
                              truncation=True,
                              max_length=MAX_SEQUENCE_LENGTH)
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(
            tf.convert_to_tensor(encoded_input[key]), 0)
    model_name = "minilm_tf_inference"
    minilm_mlir = shark_load(model_name, load_args.download_mlir_path)
    test_input = (encoded_input["input_ids"], encoded_input["attention_mask"],
         encoded_input["token_type_ids"])
    shark_module = SharkInference(
        minilm_mlir, test_input, benchmark_mode=True)
    shark_module.set_frontend("mhlo")
    shark_module.compile()
    shark_module.benchmark_all(test_input)
