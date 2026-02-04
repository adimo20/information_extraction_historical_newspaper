import dspy
from dspy import GEPA
from src.dspy_programm import Extractor
from src.optimization_metric import metric
from src.generate_input_data import DataLoaderDspy
from datetime import datetime
import os
import pickle


class AutomaticPromptOptimizer:
    """Implementation of a worklow that optimizes a dspy programm for information extraction"""
    def __init__(self, path_to_annotations:str, api_key:str):
        
        self.path_to_annotations = path_to_annotations
        self.lm = None
        self.api_key = api_key
        self.optimized_programm = None
        self.optimizer = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.model_name = "gemini/gemini-2.5-flash"

    def get_train_test_split(self):

        """
        Performs a train, val, test split on the human annotated data. The data needs to be in a specific json format, defined by label studio,
        for other adaptations of the class, the train test split method would need to be updated. Training data is saved into: data/datsets/current_data.
        The data will then be used for the optimization. Requires path_to_annoations to set.
        
        Paramters:
            None
        Returns:
            train_set:list[dspy.Example],
            val_set:list[dspy.Example],
            test_set:list[dspy.Example]

        """

        outdir = "data/"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        final_dir = os.path.join(outdir, str(datetime.now())[0:10])
        if not os.path.isdir(final_dir):
            os.mkdir(final_dir)

        load = DataLoaderDspy(path_to_annotations=self.path_to_annotations)
        train_set, val_set, test_set = load.train_test_split()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        with open(os.path.join(final_dir,'train.pkl'), 'wb') as file:
            pickle.dump(train_set, file)
        with open(os.path.join(final_dir,'test_set.pkl'), 'wb') as file:
            pickle.dump(test_set, file)
        with open(os.path.join(final_dir,'val_set.pkl'), 'wb') as file:
            pickle.dump(val_set, file)

        return train_set, val_set, test_set
    
    def initialize_model(self):

        """
        Initialises the dspy lm, that will later perform the prompt tuning, requires API. In case there are to many extractions,
        the max tokens needs to be adjusted. For reproducability the temperature is 0. 

        Parameters:
            None
        Returns:
            None - only sets the lm as part of the init

        """


        self.lm = dspy.LM(
            #maybe use light version
            self.model_name,
            api_key=self.api_key,
            temperature=0,
            max_tokens=32000
            )
        dspy.configure(lm=self.lm)
        return

    def evalutate_dspy(self, test_set):

        """
        Evaluates the final performance of the tuned prompt on the test set.

        Parameters:
            test_set:list[dspy.Example]

        Returns:
            dspy Evaluation results

        """

        return dspy.Evaluate(
            devset=test_set,
            metric=metric,
            num_threads=32,
            display_table=True,
            display_progress=True
            )

    def initial_evaluation_on_test_set(self, test_set):
        
        """
        Evaluates the performance of the initial prompt on the test set. To see if we get improvment through prompt tuning.

        Parameters:
            test_set:list[dspy.Example]

        Returns:
            dspy Evaluation results

        """


        evaluate = self.evalutate_dspy(test_set)
        program = Extractor()
        evaluate(program)
        return
    

    def optimize_prompt(self):

        """
        Workflow for the prompt otimization. At first the model is initialised and a train test split is performed. The the gepa optimizer
        is initialised, with predifined parameters, they can be adjusted. A second llm interface ist defined as evaluation model, because 
        generally a smaller llm is used to perform cheap inference and then a bigger model reflects on the used prompts.
        
        :param self: Description
        """

        self.initialize_model()

        train_set, val_set, test_set = self.get_train_test_split()

        self.initial_evaluation_on_test_set(test_set=test_set)
        
        self.optimizer = GEPA(
            metric=metric,
            auto="light",
            # To save costs and to provide overfitting - due to limited sample size reduce number of full evaluations
            # by default over 90 something.
            num_threads=32,
            track_stats=True,
            use_merge=False,
            reflection_lm=dspy.LM(
                #For reflection of the prompt use a strong model
                model="gemini/gemini-2.5-pro",
                num_retries=10,
                temperature=1.0,
                max_tokens=32000,
                api_key=self.api_key)
        )

        programm = Extractor()

        optimized_program = self.optimizer.compile(
            programm,
            trainset=train_set,
            valset=val_set
        )

        self.optimized_programm = optimized_program
        
        evaluate = self.evalutate_dspy(test_set)
      
        evaluate(self.optimized_programm)

        return self.optimized_programm

