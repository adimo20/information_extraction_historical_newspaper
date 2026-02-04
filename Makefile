PYTHON = python  #if you want to use different venvs for different tasks set the path to your venv/bin/pythonXXX here
API_KEY = $(GEMINI_API_KEY)
MODEL = gemini/gemini-2.5-flash

create-venv:
	sh create_venv.sh 

tune-prompt:
	python PromptTuning/tune_prompt.py \
	 --annotated_file data/annotations.json \
	 --key $(API_KEY)

predict-txt-file:
	python Inference_InformationExtraction/predict_from_txt.py \
	 --load_programm_path optimized_extraction \
	 --key $(API_KEY) \
	 --model_name $(MODEL)\
	 --temperatur 0 \
	 --file fulltext.txt