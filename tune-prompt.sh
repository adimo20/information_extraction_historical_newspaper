source venv_dspy/Scripts/activate
python PromptTuning/tune_prompt.py \
    --annotated_file ./PromptTuning/data/annotations.json \
	--key $GEMINI_API_KEY