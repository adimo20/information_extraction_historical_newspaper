from rapidfuzz import fuzz
import ast

#For a reference inmplementation of a similair use case see the example on the dspy website
#https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/

THRESHOLD = 0.95

def extraction_quality_precision(extractions:list[str], answer:list[str])-> float:

    """
    Calculates the precision of the extractions. The condition, if we call an llm extraction a Positive/correctly annotated sample,
    is that if we find an annotation, that meets the condition stringsimiliarity(a,b) > Treshold. 

    Parameters:
        extractions:list[str]= List of llm extractions
        answer:list[str]= List of human annoations that represent the ground truth

    Returns:
        precision:float

    """

    result = []
    for answ in answer:
        result.append(any([fuzz.ratio(answ, extraction)/100 > THRESHOLD for extraction in extractions]))
    recall = sum(result)/len(extractions)
    return recall


def extraction_quality_recall(extractions:list[str], answer:list[str])-> float:
    """
    Calculates the recall of the extractions. The condition, if we call an llm extraction a Positive/correctly annotated sample,
    is that if we find an annotation, that meets the condition stringsimiliarity(a,b) > Treshold. 

    Parameters:
        extractions:list[str]= List of llm extractions
        answer:list[str]= List of human annoations that represent the ground truth

    Returns:
        recall:float

    """
    result = []
    for extraction in extractions:
            result.append(any([fuzz.ratio(answ, extraction)/100 > THRESHOLD for answ in answer]))
    precision = sum(result)/len(answer)
    return precision


def metric(example, pred) -> float:

    """
    Calculates the success metric for the gepa optimizer. The success metric is the recall, this method could also be
    modified to optimize for precision or recall, simply by returning the value for the given metric calculated ealier 
    inside of the function. Both inputs have to be read by ast.

    Parameters:
        example:str: example is the manual annotations by the human coders
        pred: infernece given by the dspy Extractor  

    Returns:
        f1_score:float
       
    """

    manual_annotations = ast.literal_eval(example["answer"])
    exctraction_results =ast.literal_eval(pred.items()[0][1])

    if len(manual_annotations) == 0:
         if len(exctraction_results) == 0:
              return 1
         else:
             return 0
         

    recall = extraction_quality_recall(
         extractions=exctraction_results,
         answer=manual_annotations
         )
    precision =extraction_quality_precision(
         extractions=exctraction_results,
         answer=manual_annotations
         )
    
    f1_score = 2* precision*recall / (precision+recall)
    return f1_score

