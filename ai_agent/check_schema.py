import logging
import sys
import os
import mysql.connector
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('schema_check')

# Load environment variables
load_dotenv()

def check_schema():
    """Check the agent_conversations table schema"""
    try:
        # Connect to the database using credentials from .env
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
        
        cursor = connection.cursor()
        
        # Check if the table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'agent_conversations'
        """)
        
        if cursor.fetchone()[0] == 0:
            logger.error("The agent_conversations table does not exist!")
            return
        
        # Get column information
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
            FROM information_schema.COLUMNS
            WHERE TABLE_NAME = 'agent_conversations'
            ORDER BY ORDINAL_POSITION
        """)
        
        columns = cursor.fetchall()
        logger.info("Table agent_conversations columns:")
        for column in columns:
            logger.info(f"  {column[0]:<20} {column[1]:<10} {'NULL' if column[2] == 'YES' else 'NOT NULL'} {column[3]}")
            
        # Check if the visualization column exists
        has_visualization = any(col[0] == 'visualization' for col in columns)
        if has_visualization:
            logger.info("✅ The visualization column exists in the table")
        else:
            logger.error("❌ The visualization column is missing from the table!")
        
        # Close the connection
        cursor.close()
        connection.close()
        
    except Exception as e:
        logger.error(f"Error checking schema: {str(e)}")
        
if __name__ == "__main__":
    check_schema() 