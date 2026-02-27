"""Upload marketing campaign data to MongoDB"""
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def upload_data():
    """Upload CSV data to MongoDB"""
    
    # Get MongoDB URL
    mongo_url = os.getenv("MONGO_DB_URL")
    if not mongo_url:
        raise Exception("MONGO_DB_URL not found in .env file")
    
    print("Connecting to MongoDB...")
    client = MongoClient(mongo_url)
    db = client['customermind']
    collection = db['customer_segmentation']  # Match the collection name in database.py
    
    # Check if data already exists
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print(f"Database already has {existing_count} records.")
        response = input("Do you want to delete and re-upload? (yes/no): ")
        if response.lower() == 'yes':
            collection.delete_many({})
            print("Deleted existing data.")
        else:
            print("Keeping existing data. Exiting.")
            return
    
    # Read CSV
    print("Reading CSV file...")
    df = pd.read_csv('notebooks/marketing_campaign.csv', sep='\t')
    
    print(f"Found {len(df)} rows in CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Convert to dict and upload
    print("Uploading to MongoDB...")
    records = df.to_dict('records')
    collection.insert_many(records)
    
    print(f"✅ Successfully uploaded {len(records)} records to MongoDB!")
    print(f"Database: customermind")
    print(f"Collection: customer_segmentation")
    
    # Verify
    count = collection.count_documents({})
    print(f"Verification: {count} documents in collection")

if __name__ == "__main__":
    upload_data()
