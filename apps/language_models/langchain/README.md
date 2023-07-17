# Langchain

## How to run the model

1.) Install all the dependencies by running:
```shell
pip install -r apps/language_models/langchain/langchain_requirements.txt
```
2.) Create a folder named `user_path` and all your docs into that folder.
Now, you are ready to use the model.

3.) To run the model, run the following command:
```shell
python apps/language_models/langchain/gen.py --user_path=<path_to_user_path_directory> --cli=True
```
