import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'conf', '.env'))

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'gaming_mental_health_10M_40features.csv')
TABLE_NAME = 'tbl_gaming_mental_health'
CHUNK_SIZE = 10000

def main():
    engine = create_engine(
        f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    )

    total = 0
    for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)):
        chunk.to_sql(
            name=TABLE_NAME,
            con=engine,
            if_exists='append',
            index=False
        )
        total += len(chunk)
        print(f'  {total:,} rows inserted...')

    print(f'\n완료: 총 {total:,} rows')

if __name__ == '__main__':
    main()
