"""Script to delete Qdrant collection.

Usage:
    python delete_collection.py
    or
    .\venv\Scripts\python.exe delete_collection.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qdrant_client import QdrantClient
    from app.core.config import settings
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure you're using the virtual environment:")
    print("  .\\venv\\Scripts\\activate")
    print("  python delete_collection.py")
    sys.exit(1)

def delete_collection():
    """Delete the Qdrant collection."""
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collection_name = settings.qdrant_collection
        
        print(f"Connecting to Qdrant at {settings.qdrant_url}...")
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            points_count = collection_info.points_count
            vector_size = collection_info.config.params.vectors.size
            
            print(f"\nCollection '{collection_name}' found:")
            print(f"  - Points: {points_count}")
            print(f"  - Vector size: {vector_size}")
            
            if points_count > 0:
                print(f"\n⚠ WARNING: Collection has {points_count} points!")
                print("This will delete all data in the collection.")
                response = input("Are you sure you want to delete it? (type 'yes' to confirm): ")
                if response.lower() != "yes":
                    print("Deletion cancelled.")
                    return
            
            # Delete collection
            print(f"\nDeleting collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print(f"✓ Collection '{collection_name}' deleted successfully!")
            print("\nYou can now restart the server and it will create a new collection with the correct vector size (768 for nomic-embed-text).")
            
        except Exception as e:
            error_str = str(e).lower()
            if "doesn't exist" in error_str or "not found" in error_str or "does not exist" in error_str:
                print(f"Collection '{collection_name}' does not exist.")
            else:
                raise
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    delete_collection()

