import argparse
from TunedExtraction import InformationExtractor


parser = argparse.ArgumentParser()
parser.add_argument("--load_programm_path", help="Location of the annotated dataset")
parser.add_argument("--key", help="API-Key")
parser.add_argument("--model_name", help="Modelname for inference")
parser.add_argument("--temperature", help="Temperature of generation")
parser.add_argument("--file", help="File where we want to extract information from")

args = parser.parse_args()


if __name__ == "__main__":

    print(f"Loadeding dspy-Model {args.load_programm_path}")
    extractor = InformationExtractor(
        load_programm_path=args.load_programm_path,
        model_name=args.model_name,
        API_KEY=args.key,
        temperatur=float(args.temperature))
    print("Loading dspy-Model done!")
    print(f"Opening file {args.file} for extraction!")
    with open(args.file, "r", encoding="utf") as f:
        page = f.read()
    print("Running Inference!")
    print(extractor.predict(page))
   