from src.optimizer_dspy import AutomaticPromptOptimizer
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--annotated_file", help="Location of the annotated dataset")
parser.add_argument("--key", help="Location of the annotated dataset")
args = parser.parse_args()


optim = AutomaticPromptOptimizer(
    path_to_annotations=args.annotated_file,
    api_key=args.key
)

print("Starting prompt tuning!")

optimized_programm = optim.optimize_prompt()

for name, pred in optim.optimized_programm.named_predictors():
    with open("optimized_prompt.txt", "w", encoding="utf-8") as f:
        f.writelines(pred.signature.instructions)

optim.optimized_programm.save("./optimized_extraction_" + str(datetime.datetime.now()).split(" ")[0] + "/", save_program=True)