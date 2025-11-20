import numpy as np
import json
from random import (
    randint,
    seed as seed_random,
    getstate as random_getstate,
    setstate as random_setstate,
)


# Generate and return a new seed if the provided one is not in the
# supported range (including -1)
def sanitize_seed(seed: int | str):
    seed = int(seed)
    uint32_info = np.iinfo(np.uint32)
    uint32_min, uint32_max = uint32_info.min, uint32_info.max
    if seed < uint32_min or seed >= uint32_max:
        seed = randint(uint32_min, uint32_max)
    return seed


# take a seed expression in an input format and convert it to
# a list of integers, where possible
def parse_seed_input(seed_input: str | list | int):
    if isinstance(seed_input, str):
        try:
            seed_input = json.loads(seed_input)
        except (ValueError, TypeError):
            seed_input = None

    if isinstance(seed_input, int):
        return [seed_input]

    if isinstance(seed_input, list) and all(type(seed) is int for seed in seed_input):
        return seed_input

    raise TypeError(
        "Seed input must be an integer or an array of integers in JSON format"
    )


# Generate a set of seeds from an input expression for batch_count batches,
# optionally using that input as the rng seed for any randomly generated seeds.
def batch_seeds(seed_input: str | list | int, batch_count: int, repeatable=False):
    # turn the input into a list if possible
    seeds = parse_seed_input(seed_input)

    # slice or pad the list to be of batch_count length
    seeds = seeds[:batch_count] + [-1] * (batch_count - len(seeds))

    if repeatable:
        if all(seed < 0 for seed in seeds):
            seeds[0] = sanitize_seed(seeds[0])

        # set seed for the rng based on what we have so far
        saved_random_state = random_getstate()
        seed_random(str([n for n in seeds if n > -1]))

    # generate any seeds that are unspecified
    seeds = [sanitize_seed(seed) for seed in seeds]

    if repeatable:
        # reset the rng back to normal
        random_setstate(saved_random_state)

    return seeds
