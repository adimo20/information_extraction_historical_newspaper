import pandas as pd
from ddbapi import zp_pages
import random

# Newspapers and corresponding ZDB IDs:
# - General-Anzeiger, unabhängige Tageszeitung für Bonn ; Bonner Stadtanzeiger
#       ZDB-ID: 2815866-0
# - Hamburger Fremdenblatt
#       ZDB-ID: 3024925-9
# - Münchner neueste Nachrichten, Wirtschaftsblatt, alpine und Sport-Zeitung, Theater- und Kunst-Chronik
#       ZDB-ID: 3136538-3
# - Badische Presse, Generalanzeiger der Residenz Karlsruhe und des Großherzogtums Baden
#       ZDB-ID: 2797055-3

default_zdb_ids = [
    "2815866-0",
    "3024925-9",
    "3136538-3",
    "2797055-3"
]

class DataCollector:
    def __init__(self, zdb_ids:list[str], write_output:bool=False, output_path:str=None, query:list[str]=None):
        self.zdb_ids: list[str] = default_zdb_ids if zdb_ids is None else zdb_ids
        self.write_output = write_output
        self.output_path = output_path
        self.query = query
        self.retrieved_data = None

    def save_data(self)->None:
        """Write a csv-file containing the retrieved data, the output file will be stored in the place set in output path
        Parameters:
            None
        Returns:
            None
        """
        if self.write_output:
            self.retrieved_data.to_csv(self.output_path)
        return
        
    def get_data_from_query(self) -> pd.DataFrame:

        """
        Loads data from ddbapi and saves it as a pandas dataframe object into self.retrieved_data, the retrieved data
        can be defined by query and places, that are part of the init. Due to the api beeing slow and to timeout errors 
        while receiving the data, data is retrieved through a loop.

        Parameters:
            None
        Returns:
            pd.Dataframe

        """

        df_list = []
        for q in self.query:
            for zdb_id in self.zdb_ids:
                df = zp_pages(
                    publication_date='[1850-01-01T12:00:00Z TO 1980-12-31T12:00:00Z]', 
                    zdb_id=zdb_id,
                    plainpagefulltext=q
                    )
                if df is not None and len(df) > 0:
                    df["query"] = q
                    df["zdb_id"] = zdb_id
                    df_list.append(df)
        

        if len(df_list) > 1:
            self.retrieved_data = pd.concat(df_list)
            self.retrieved_data.drop_duplicates(subset="plainpagefulltext", inplace=True)
            self.save_data()
        else:
            self.retrieved_data = df_list[0]
            self.retrieved_data.drop_duplicates(subset="plainpagefulltext", inplace=True)
            self.save_data()
        return self.retrieved_data
    
    def create_random_sample(self, n:int) -> pd.DataFrame:

        """Takes a random sample of n examples from the retrieved dataframe"""
        random.seed(42)
        return self.retrieved_data.sample(n=n, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    
    collection = DataCollector(
        zdb_ids=default_zdb_ids,
        write_output=False,
        query=["zwecks heirat"]
        )
    retrieved_data = collection.get_data_from_query()