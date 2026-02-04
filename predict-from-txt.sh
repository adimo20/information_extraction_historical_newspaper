source venv_dspy/Scripts/activate
python PromptTuning/predict_from_txt.py \
	 --load_programm_path ./PromptTuning/optimized_extraction/ \
	 --key $GEMINI_API_KEY \
	 --model_name gemini/gemini-2.5-flash \
	 --temperature 0 \
	 --file ./PromptTuning/examples/fulltext.txt