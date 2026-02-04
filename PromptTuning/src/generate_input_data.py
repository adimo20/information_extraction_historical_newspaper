import dspy
import json
import random

#https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/

# Creates a custom dataset for our use-case. That can be processed by the extractor defined in dspy_programm.py and the optimization_metric.py

class DataLoaderDspy:

    def __init__(self, path_to_annotations):

        self.path = path_to_annotations

    def load_annotations(self):

        """
        Loads annotated data from label studio. The annotations need to be in json-format.
        Path to the annotations is received thorugh the init
        
        Returns:
            annotated_data:dict
        
        """

        with open(self.path, "r", encoding="utf-8") as f:
            annotated_data = json.loads(f.read())
        
        return annotated_data

    def generate_ideal_output(self):
        
        """
        Generate Gold-Standard Examples for the measurement of the quality of the information extraction. All extractions, that come from 
        the llm, will be evaluated on the bases of those examples.

        Loads the annotations and generates two sorts of informations - the input text for the llm and the corresponding gold predictions.

        Parameters:
            None

        Returns:
            input_text:list[str]
            ideal_output:list[str]         

        """

        annotated_data = self.load_annotations()
        ideal_output = [
            
                [
                    annotated_data[j]["annotations"][0]["result"][i]["value"]["text"] 
                    for i in range(len(annotated_data[j]["annotations"][0]["result"]))
                ]    for j in range(len(annotated_data))
            ]

        input_text = [
            annotated_data[i]["data"]["text"] for i in range(len(annotated_data))
            ]


        return input_text, ideal_output
    

 
    def generate_entry(self, initial_text:str, gold_extraction:str)-> dict:

        """Generates an entry for the dspy dataset. It is important that the mesage and answer format as defined here is keept, because this makes
        it readable for the dspy module and better distinuish. 
        That has the keys message and answer, as the dspy GEPA optimizer needs.
        
        Parameters: 
            initial_text:str
            gold_extraction:str
        
        Returns:
            dict[str,str]
        
        """


        return {
            'message': initial_text
            ,
            'answer': json.dumps(gold_extraction, ensure_ascii=False)                
        }

    def generate_dspy_dataset(self):

        """
        Generates an DSPY dataset with entry, that have the datatype dspy example.   
        
        Parameters:
            None
        Returns:
            dspy_dataset:list[dspy.Example] -> ready to be used by prompt tuning

        """

        input_text, ideal_output = self.generate_ideal_output()

        dspy_dataset = [
            dspy.Example(
                self.generate_entry(initial_text=input_text[i], gold_extraction=ideal_output[i])
                ).with_inputs("message") for i in range(len(ideal_output))
        ]

        return dspy_dataset
    
    def train_test_split(self):

        """Performs the traintest split, where every element contains is a dspy example, 
        splits into train, val and test-set
        
        Parameters:
            None
        
        Returns:
            train_set:list[dspy.Example],
            val_set:list[dspy.Example],
            test_set:list[dspy.Example]


        """

        dspy_dataset = self.generate_dspy_dataset()
        random.Random(0).shuffle(dspy_dataset)
        train_set = dspy_dataset[:int(len(dspy_dataset) * 0.5)]
        val_set = dspy_dataset[int(len(dspy_dataset) * 0.5):int(len(dspy_dataset) * 0.75)]
        test_set = dspy_dataset[int(len(dspy_dataset) * 0.75):]

        return train_set, val_set, test_set

