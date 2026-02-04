# Extracting marriage requests from historic newspapers using llms

## Table of content


## Description and use case

## Project Structure

```plaintext
├── DataCollection/
│   ├── DataCollector.py
│   └── requirements.txt
├── PromptTuning/
│   ├── data/
│   │   └── annotations.json
│   ├── examples/
│   │   └── fulltext.txt
│   ├── optimized_extraction/
│   │   ├── metadata.json
│   │   └── program.pkl
│   ├── src/
│   │   ├── config.py
│   │   ├── dspy_programm.py
│   │   ├── generate_input_data.py
│   │   ├── optimization_metric.py
│   │   └── optimizer_dspy.py
│   ├── predict_from_text.py
│   ├── tune_prompt.py
│   ├── TunedExtraction.py
│   └── requirements.txt
├── data/
│   └── 2026-01-28/
│       ├── test_set.pkl
│       ├── train.pkl
│       └── val_set.pkl
├── Makefile
├── README.md
├── create_venv_data_collection.sh
├── create_venv_dspy.sh
├── predict-from-txt.sh
└── tune-prompt.sh
```

## Application

### Virtual environment

To run the prompt-tuning and the inference in this project you need to at first set up a virtual environment which will be used to run the code (conda  works as well - but you would need to create your conda env yourself). To do this run:

```sh
sh create_venv_dspy.sh
```

or for the data collection:

```sh
sh create_venv_data_collection.sh
```

### Prompt tuning

To start a prompt tuning job you can run a shell script. The shell script at first activates the virtual environment from where the code is exceuted. The it sources the python file which runs the prompt tuning. The python script takes two arguments as input (via argparse). The first argument --annotated_file where the path to the annotated datset needs to be specified. The datset needs to be in the format of a label studio json-output, so the the input for the prompt optimizer can be produced. The second argument is --key, where the API-key for the gemini api needs to be specified. For convinience we recommend storing the key in an environment variable GEMINI_API_KEY. Alternativly if you want to source the prompt tuning from a notebook you can take the file PromptTuning/tune_prompt.py as reference and modify it how you need it.

```sh
source venv_dspy/Scripts/activate
python PromptTuning/tune_prompt.py \
    --annotated_file ./PromptTuning/data/annotations.json \
	--key $GEMINI_API_KEY
```


To extract information/marriage requests form a txt-file you can run from the terminal as follows:

```sh
source venv_dspy/Scripts/activate
python PromptTuning/predict_from_txt.py \
	 --load_programm_path ./PromptTuning/optimized_extraction/ \
	 --key $GEMINI_API_KEY \
	 --model_name gemini/gemini-2.5-flash \
	 --temperature 0 \
	 --file ./PromptTuning/examples/fulltext.txt
```



## Data Source

The deutsche digitale bibliothek offers with its digital newspaper collection access to over 600000 historic newspapers issues ranging from the year 1671 to 1950 with over 4.5 Mio pages (<a href="https://dbis.ur.de/DM/resources/104835">see Deutsches Zeitungsportal</a>). From those around 82% of the issues can searched via fulltext search, what makes it possible to identify certain newspaper pages, where a specific key word is mentioned. Those pages are ocr read, what makes it possible to not only identify pages, where a certain word is mentioned, but to further extract more detailed and condensed information.
The pages, including their ocr-read texts, incl. metadata can retrieved via <a href="https://github.com/Deutsche-Digitale-Bibliothek/ddblabs-ddbapi">API</a>.

## Data Collection

The data for the following project is retrieved via the `ddbapi` python wrapper, which is provided by the deutsche digitale bibiliothek on github/pypi. The script DataCollection/DataCollector.py implements the class `DataCollector`, which retrieves the data from the `ddbapi` and saves it into a pandas dataframe. Due to rate limits/time-outs inside the api, we have to collect the data iterativly for every query and place. If we would call the list of queries and places all at once the api connection would close after a certain amout of time due to the dataset we will retrieve beeing to big big in most cases. 

```py

from DataCollection.DataCollector import DataCollector

collection = DataCollector(
        places=["Hamburg", "München"],
        write_output=False,
        query=["zwecks Heirat", "zwecks heirat"]
        )
retrieved_data = collection.get_data_from_query()

```

## Prompt Tuning

## Information Extraction


