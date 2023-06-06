from apps.language_models.src.pipelines.vicuna_pipeline import Vicuna

model = Vicuna("vicuna", device="cpu")
first = model.compile_first_vicuna()

second = model.compile_second_vicuna()
