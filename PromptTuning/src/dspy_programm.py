import dspy
from dspy import GEPA
from src.config import PROMPT_1


class ExtractionSignature(dspy.Signature):
    """
    Defines how the information extraction should be done - especially that there will be an initial prompt.
    Paramters:
        None
    """
    
    message = dspy.InputField(desc=PROMPT_1)
    
    extractions = dspy.OutputField()

class Extractor(dspy.Module):

    """
    Extractor class will be used to extract the informations from the text we give a input.
    Extractor will also be used to tune the prompt. The forward function is the actual prediction/inference function.
    Besides this very simple code everything es is done, by the dspy Module.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ExtractionSignature)

    def forward(self, message):
        return self.predictor(message=message)

