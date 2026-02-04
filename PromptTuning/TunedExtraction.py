import dspy
import ast

class InformationExtractor:

    def __init__(self, load_programm_path, model_name, API_KEY, temperatur):
        
        self.model_name = model_name
        self.api_key = API_KEY
        self.temperature = temperatur
        self.load_programm_path = load_programm_path 
        self.initialise_lm()
        self.load_programm()
    
    def initialise_lm(self) -> None:
        self.lm = dspy.LM(
            self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=32000
            )
        dspy.configure(lm=self.lm)
    
    def load_programm(self) -> None:
        self.programm = dspy.load(self.load_programm_path)
    
    def predict(self, fulltext)->list[str]:
        return ast.literal_eval(self.programm(fulltext).extractions)
