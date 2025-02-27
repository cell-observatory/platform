import os
import tensorstore as ts
from supabase import create_client, Client
from dotenv import load_dotenv

class FishDatabase:
    """
    A class to access the preprocessed dataset and metadata.
    """
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")

        # connect to the database
        self.supabase: Client = create_client(url, key)

    def __len__(self):
        return 0 # TODO

    def __getitem__(self, index):
        pass # TODO