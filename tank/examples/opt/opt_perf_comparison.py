import collections
import json
import time
import os

from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from shark_opt_wrapper import OPTForCausalLMModel

MODEL_NAME = "facebook/opt-1.3b"
OPT_MODELNAME = "opt-1.3b"
OPT_FS_NAME = "opt_1-3b"
MAX_SEQUENCE_LENGTH = 128
DEVICE = "cpu"

PROMPTS8 = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]
PROMPTS128 = [
    "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes",
    "I wish either my father or my mother, or indeed both of them, as they were in duty both equally bound to it, had minded what they were about when they begot me; had they duly consider’d how much depended upon what they were then doing;—that not only the production of a rational Being was concerned in it, but that possibly the happy formation and temperature of his body, perhaps his genius and the very cast of his mind;—and, for aught they knew to the contrary, even the fortunes of his whole house might take their turn from the humours and dispositions which were then uppermost;—Had they duly weighed and considered all this, and proceeded accordingly,—I am verily persuaded I should have made a quite different figure",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only. You may think this sentence too long, but I",
    "We may give what explanation we please of this unwillingness; we may attribute it to pride, a name which is given indiscriminately to some of the most and to some of the least estimable feelings of which mankind are capable; we may refer it to the love of liberty and personal independence, as appeal to which was with the Stoics one of the most effective means for the inculcation of it; to the love of power or to the love of excitement, both of which do really enter into and contribute to it; but its most appropriate appellation is a sense of dignity, which all human beings possess in one form or other, and in some, though by no means in exact, proportion to their higher faculties",
    "But then they were married (she felt awful about being pregnant before but Harry had been talking about marriage for a while and anyway laughed when she told him in early February about missing her period and said Great she was terribly frightened and he said Great and lifted her put his arms around under her bottom and lifted her like you would a child he could be so wonderful when you didn’t expect it in a way it seemed important that you didn’t expect it there was so much nice in him she couldn’t explain to anybody she had been so frightened about being pregnant and he made her be proud) they were married after her missing her second period in March and she was still",
    "Considering how common illness is, how tremendous the spiritual change that it brings, how astonishing, when the lights of health go down, the undiscovered countries that are then disclosed, what wastes and deserts of the soul a slight attack of influenza brings to light, what precipices and lawns sprinkled with bright flowers a little rise of temperature reveals, what ancient and obdurate oaks are uprooted in us in the act of sickness, how we go down into the pit of death and feel the waters of annihilation close above our heads and wake thinking to find ourselves in the presence of the angels and the harpers when we have a tooth out and come to the surface in the dentist’s arm chair and confuse his ‘Rinse the mouth—",
    "Just exactly like Father if Father had known as much about it the night before I went out there as he did the day after I came back thinking Mad impotent old man who realized at last that there must be some limit even to the capabilities of a demon for doing harm, who must have seen his situation as that of the show girl, the pony, who realizes that the principal tune she prances to comes not from horn and fiddle and drum but from a clock and calendar, must have seen himself as the old wornout cannon which realizes that it can deliver just one more fierce shot and crumble to dust in its own furious blast and recoil, who looked about upon the scene",
    "She said I’m tired of begging God to overthrow my son, because all this business of living in the presidential palace is like having the lights on all the time, sir, and she had said it with the same naturalness with which on one national holiday she had made her way through the guard of honor with a basket of empty bottles and reached the presidential limousine that was leading the parade of celebration in an uproar of ovations and martial music and storms of flowers and she shoved the basket through the window and shouted to her son that since you’ll be passing right by take advantage and return these bottles to the store on the corner, poor mother. What a long sentence, don't you think",
    "Sometimes, though, there is a ghostly rumble among the drums, an asthmatic whisper in the trombones that swings me back into the early twenties when we drank wood alcohol and every day in every way grew better and better, and there was a first abortive shortening of the skirts, and girls all looked alike in sweater dresses, and people you didn’t want to know said ‘Yes, we have no bananas’, and it seemed only a question of a few years before the older people would step aside and let the world be run by those who saw things as they were and it all seems rosy and romantic to us who were young then, because we will never feel quite so intensely about our surroundings any more",
    "All round them, ten, scores, it seems like hundreds, of faces and bodies are perspiring, trooping and bellying up the stairs with arterio-sclerotic grimaces past a showcase full of such novel items as Joy Buzzers, Squirting Nickels, Finger Rats, Scary Tarantulas and spoons with realistic dead flies on them, past Fred’s barbershop, which is just off the landing and has glossy photographs of young men with the kind of baroque haircuts one can get in there, and up onto 50th Street into a madhouse of traffic and shops with weird lingerie and gray hair-dyeing displays in the windows, signs for free teacup readings and a pool-playing match between the Playboy Bunnies and Downey’s Showgirls, and then everybody pounds on toward the Time-Life Building, the Brill Building or",
]

if MAX_SEQUENCE_LENGTH > 8:
    PROMPTS = PROMPTS128

ModelWrapper = collections.namedtuple("ModelWrapper", ["model", "tokenizer"])


def create_vmfb_module(model_name, tokenizer, device):
    opt_base_model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        "What is the meaning of life?",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])

    mlir_path = f"./{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch.mlir"
    if os.path.isfile(mlir_path):
        with open(mlir_path, "r") as f:
            model_mlir = f.read()
        print(f"Loaded .mlir from {mlir_path}")
    else:
        (model_mlir, func_name) = import_with_fx(
            model=opt_model,
            inputs=inputs,
            is_f16=False,
            model_name=OPT_FS_NAME,
            return_str=True,
        )
        with open(mlir_path, "w") as f:
            f.write(model_mlir)
        print(f"Saved mlir at {mlir_path}")

    shark_module = SharkInference(
        model_mlir,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )

    vmfb_name = f"{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}_tiled_ukernels"
    shark_module.save_module(module_name=vmfb_name)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def load_shark_model() -> ModelWrapper:
    vmfb_name = f"{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}_tiled_ukernels.vmfb"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if not os.path.isfile(vmfb_name):
        print(f"vmfb not found. compiling and saving to {vmfb_name}")
        create_vmfb_module(OPT_MODELNAME, tokenizer, DEVICE)
    shark_module = SharkInference(mlir_module=None, device="cpu-task")
    shark_module.load_module(vmfb_name)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, prompt: str):
    model_inputs = model_wrapper.tokenizer(
        prompt,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )
    # Generate logits output of OPT model.
    return model_wrapper.model("forward", inputs)


def run_shark():
    model_wrapper = load_shark_model()

    prompt = "What is the meaning of life?"
    logits = run_shark_model(model_wrapper, prompt)

    # Print output logits to validate vs. pytorch + base transformers
    print(logits[0])


def load_huggingface_model() -> ModelWrapper:
    return ModelWrapper(
        model=OPTForCausalLM.from_pretrained(MODEL_NAME),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )


def run_huggingface_model(model_wrapper: ModelWrapper, prompt: str):
    inputs = model_wrapper.tokenizer(prompt, return_tensors="pt")
    return model_wrapper.model.forward(
        inputs.input_ids, inputs.attention_mask, return_dict=False
    )


def run_huggingface():
    model_wrapper = load_huggingface_model()

    prompt = "What is the meaning of life?"
    logits = run_huggingface_model(model_wrapper, prompt)

    print(logits[0])


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


def collect_huggingface_logits():
    t0 = time.time()
    model_wrapper = load_huggingface_model()
    print("--- Took {} seconds to load Huggingface.".format(time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print("prompt: {}".format(prompt))
        logits = run_huggingface_model(model_wrapper, prompt)
        results.append([prompt, logits[0].tolist()])
    print("--- Took {} seconds to run Huggingface.".format(time.time() - t0))
    save_json(results, "/tmp/huggingface.json")


def collect_shark_logits():
    t0 = time.time()
    model_wrapper = load_shark_model()
    print("--- Took {} seconds to load Shark.".format(time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print("prompt: {}".format(prompt))
        logits = run_shark_model(model_wrapper, prompt)
        lst = [e.tolist() for e in logits]
        results.append([prompt, lst])
    print("--- Took {} seconds to run Shark.".format(time.time() - t0))
    save_json(results, "/tmp/shark.json")


if __name__ == "__main__":
    collect_shark_logits()
    collect_huggingface_logits()
