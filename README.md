# Extracting marriage requests from historic newspapers using llms

## Table of content

- [Description and use case](#description-and-use-case)
- [Project Structure](#project-structure)
- [Application](#application)
    - [Virtual environment](#virtual-environment)
    - [Prompt tuning](#prompt-tuning)
    - [Marriage request extraction](#marriage-request-extraction)
- [Data Source](#data-source)
- [Implementation details](#implementation-details)
    - [Data Collection](#data-collection)
    - [Prompt Tuning](#prompt-tuning-1)

## Description and use case

This project addresses the challange of automatically extracting structured information from german historical newspaper pages - more specifically extracting **marriage requests**.

Historical newspapers contain a broad range of information about social phenomenons hidden in unstructured OCR-read text. Marriage requests contain usually a very predictable set of informations, which helps us (and the llm) to specifically extract those. Those informations are: the requester's name, age, occupation, place of residence, desired partner characteristics, and sometimes a contact address. Howevery, because the texts we work with comes from very old prints, that are often very noisy, inconsitently formatted and mixed with lots of unrelated content on the same page - rule-based information extraction would hit its limits. Thus we propose a llm-based pipeline to reliably extract structured information from those unstructured texts. 
The goals of the pipeline are:

1. **Collecting relevant newspaper pages**  from the deutsches zeitungsportal, by using a keyword search via their publicly accessible API - for more information see the paragraph [Data Source](#data-source).
2. **Tuning a prompt** via the GEPA algorithm for a Large Language Model (LLM) on the task of structured information extraction, using the [DSPy](https://github.com/stanfordnlp/dspy) framework and a small set of human-annotated examples, so that the model reliably extracts structured marriage request, based on our definition of a marriage request.
3. **Run inference** using the tuned dspy progam on new unseen newspaper pages to produce structured information. 

## How do we define a marriage request ? 

Description from the seed prompt:
```txt

1. **Self-Description:** The ad must contain a brief description of the seeking person (e.g., Age/Alter, Marital Status/Familienstand, Religion/Konfession, Profession/Beruf, Wealth/Vermögen).
2. **Intent:** Marriage must be explicitly or implicitly stated as a goal or option.
   - Note: Non-binding phrases like "spätere Heirat möglich" (later marriage possible) are sufficient.
   - Keywords/Synonyms indicating intent: "Ehe", "Gatte/Gattin", "Vermählung", "Heirat", "ehelich", "Lebensgefährte" (only if marriage context is clear), "Mitgift" (dowry).
3. **Single Seeker:** The search must be for the author themselves.

1. **Business Entry (Einheirat):** Exclude ads seeking strictly to buy into or join a business/firm via marriage without a focus on the personal relationship.
2. **Third-Party Searches:** Exclude parents looking for partners for their children, or friends looking for friends.
3. **Group Searches:** Exclude ads where more than one person is searching (e.g., "Zwei junge Damen suchen...").
4. **Non-Marital:** Exclude ads asking purely for companionship ("Lebensgemeinschaft", "Gefährte") without any mention or implication of marriage.
5. **Reprints/Quotes:** Exclude citations or discussions of other marriage ads; extract only the actual ad.
6. **Non-German:** Exclude ads primarily in other languages.
```




## Project Structure

```
├── DataCollection/
│   ├── DataCollector.py          # Class to retrieve pages from the DDB API
│   └── requirements.txt          # Dependencies for the data collection
├── PromptTuning/
│   ├── data/
│   │   └── annotations.json      # Human-annotated examples in Label Studio JSON format
│   ├── examples/
│   │   └── fulltext.txt          # Sample newspaper page text for the demo inference
│   ├── optimized_extraction/
│   │   ├── metadata.json         # Metadata about the saved DSPy program - cotnaining information about version dependencies
│   │   └── program.pkl           # Tuned DSPy program in pickle format
│   ├── src/
│   │   ├── config.py             # Central configuration containing the prompt and the API-Key
│   │   ├── dspy_programm.py      # DSPy module defining the extraction signature and program
│   │   ├── generate_input_data.py# Converts Label Studio annotations into DSPy training examples
│   │   ├── optimization_metric.py# Custom metric used by DSPy optimizer to evaluate information extraction quality
│   │   └── optimizer_dspy.py     # Class that defines the DSPy GEPA-Optimizer
│   ├── predict_from_text.py      # load tuned program and run on a text file
│   ├── tune_prompt.py            # run prompt tuning on annotated data
│   ├── TunedExtraction.py        # Wrapper for loading and running the tuned program
│   └── requirements.txt          # Dependencies for prompt tuning and inference
├── data/
│   └── 2026-01-28/
│       ├── train.pkl             # Training split
│       ├── val_set.pkl           # Validation split
│       └── test_set.pkl          # Test split
├── Makefile                      
├── README.md
├── create_venv_data_collection.sh  # Sets up venv for DataCollection
├── create_venv_dspy.sh             # Sets up venv for PromptTuning and Inference
├── predict-from-txt.sh             # Shell wrapper to run prediction
└── tune-prompt.sh                  # Shell wrapper to run prompt tuning
```

## Application

There are **two seperate virtual python environments** which will be seperatly used for this project. One is used for the data collection, the other one is used for the prompt tuning and inference. This is done to resolve depency conflicts, that may occur. The main requirements for the dspy optimized programm are:

```json
{
  "dependency_versions": {
    "python": "3.13",
    "dspy": "3.0.4",
    "cloudpickle": "3.1"
  }
}
```



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

To start a prompt tuning job you can run a shell script (`tune-prompt.sh`). The shell script at first activates the virtual environment from where the code is exceuted. The it sources the python file which runs the prompt tuning. The python script takes two arguments as input (via argparse). 
    - `--annotated_file`: path to the annotated datset. The datset needs to be in the format of **a label studio json-output**, so the the input for the prompt optimizer can be produced. 
    - `--key`: API-key for the Gemini-API. For convinience we recommend storing the key in an environment variable `GEMINI_API_KEY` Alternativly if you want to source the prompt tuning from a notebook you can take the file PromptTuning/tune_prompt.py as reference and modify it how you need it.

```sh
source venv_dspy/Scripts/activate
python PromptTuning/tune_prompt.py \
    --annotated_file ./PromptTuning/data/annotations.json \
	--key $GEMINI_API_KEY
```

The tuning process can take a few moments. The progress will be printed into the console. After the process is done the tuned program will be saved into a time stamped folder called `./optimized_extraction`. From there you can load the tuned prompt/whole dspy programm later. 

**Tip** - you can use **ml_flow** to log intermediate results. 

### Marriage request extraction

To demonstrate the information/marriage request extraction form a txt-file you can as well run a shell file (`predict-from-txt.sh`). The dspy programm will be loaded and the extract the marriage requests from a given txt file. There you would also need to specify some paramters at first:    

   - `--load_programm_path`: Path where the dspy-optimized programm is stored. The folder must contain the files metadata.json and program.pkl.
   - `--key`: API-key for the Gemini-API.
   - `--model_name`: name of the model that should perform the inference, as specified in the gemini documentation.
   - `--temperature`: Temperature of the generation. Recommend is 0 for reproducability.
   - `--file`: file where we want to extract the information from.

```sh
source venv_dspy/Scripts/activate
python PromptTuning/predict_from_txt.py \
	 --load_programm_path ./PromptTuning/optimized_extraction/ \
	 --key $GEMINI_API_KEY \
	 --model_name gemini/gemini-2.5-flash \
	 --temperature 0 \
	 --file ./PromptTuning/examples/fulltext.txt
```

A sample input file is provided at `PromptTuning/examples/fulltext.txt` to test the pipeline without needing to collect new data first.


## Data Source

The deutsche digitale bibliothek offers with its digital newspaper collection access to over 600000 historic newspapers issues ranging from the year 1671 to 1950 with over 4.5 Mio pages (<a href="https://dbis.ur.de/DM/resources/104835">see Deutsches Zeitungsportal</a>). From those around 82% of the issues can searched via fulltext search, what makes it possible to identify certain newspaper pages, where a specific key word is mentioned. Those pages are ocr read, what makes it possible to not only identify pages, where a certain word is mentioned, but to further extract more detailed and condensed information.
The pages, including their ocr-read texts, incl. metadata can retrieved via <a href="https://github.com/Deutsche-Digitale-Bibliothek/ddblabs-ddbapi">API</a>.

The search terms used in this project to find marriage requests are:
- `"zwecks Heirat"` (for the purpose of marriage)
- `"zwecks heirat"` (lowercase variant, as OCR output is inconsistent)

These phrases appear characteristically in the preamble of marriage request classified ads from the late 19th and early 20th century.



## Implementation Details

### Data Collection

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

### Prompt Tuning

### Prompt Tuning Pipeline

The prompt tuning pipleine is stored under `PromptTuning/src/` and will be called from `PromptTuning/tune_prompt.py`. tune_prompt.py loads the AutomaticPromptOptimizer class and starts the tuning job. After tuning the tuned programm, as well as the produced prompt is saved. It uses argparse to parse the arguments, that are nessecary for the prompt tuning process.


#### config.py

Only stores the initial seed prompt and the API-Key. The API-Key will be loaded from an environment vaiable called `GEMINI_API_KEY` 

#### generate_input_data.py

`generate_input_data.py` implements the class `DataLoaderDspy`, which has three pourposes. It loads the annotations, form the Label Studio JSON annotation export, converts it into the ideal output format, which is required for the dspy optimization and performs a train test split and returns the train, val, test-sets.

Usage: 

```py 
from PromptTuning.src.generate_input_data import DataLoaderDspy

loader = DataLoaderDspy(path_to_annotations="Path to the annotations")
train_set, val_set, test_set = loader.train_test_split()

```

#### dspy_programm.py

Defines the core DSPy extraction module, which defines the interface on how the model receives an input and what output it should return. The current implementation is more or less raw. It could be extended with the use of Pydantic-Schemes to provide the dspy framework with more insight on how input and output should look like.

The implementation includes:

- A DSPy Signature, which defines the input and output filed of the model. The input filed is filled with the seed prompt. 
- A DSPy Module, which will handle the information extraction for us in the forward method. 

#### optimization_metric.py

Defines the evaluation metric, which is used by the DSPy optimizer to score each candidate prompt during the optimization loop. The metric we want to optimize for is the f1-score.  The key function implemeted here is metric, which calculates the numeric success shown to the gepa optimizer. This could potentially be extrended with the implementation of a metric_with_feedback, so we don't only provide the model with a numeric measure of success, but only providing a natural language feedback.


#### optimizer_dspy.py

This script implements the class `AutomaticPromptOptimizer`, which will be used to run the end-to-end process of prompt-tuning. It takes the path to the annotations and the api key as input. The flow that will be executed inside of optimize_prompt is the following. At first the dspy lm model will be initialized. The the data is loaded and a train test split is performed. After this, an intial evaluation on the test set is performed with the seed prompt to calculate the performance baseline against which we'll measure the tuned prompts performance. After this the gepa optimization will be executed. After the optimization is done we'll measure the performance of the tuned prompt und the test set to check if the newly constructed prompt performs better, that the intial one. The optimized programm will be saved in self.optimized_programm and also returned. 

Usage:

```py
from src.PromptTuning.src.optimizer_dspy import AutomaticPromptOptimizer

optim = AutomaticPromptOptimizer(
    path_to_annotations="Path to annotations,
    api_key="API_KEY here"
)

optimized_programm = optim.optimize_prompt()


```
