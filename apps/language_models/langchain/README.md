# Langchain

## How to run the model

1.) Install all the dependencies by running:
```shell
pip install -r apps/language_models/langchain/langchain_requirements.txt
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
```

2.) Create a folder named `user_path` in `apps/language_models/langchain/` directory.

Now, you are ready to use the model.

3.) To run the model, run the following command:
```shell
python apps/language_models/langchain/gen.py --cli=True
```
