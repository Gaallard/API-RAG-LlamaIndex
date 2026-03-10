"""Script to check current configuration and Qdrant collection status."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qdrant_client import QdrantClient
    from app.core.config import settings
    from app.storage.qdrant_store import DEFAULT_VECTOR_DIMENSIONS
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure you're using the virtual environment:")
    print("  .\\venv\\Scripts\\activate")
    print("  python check_config.py")
    sys.exit(1)

def check_config():
    """Check configuration and Qdrant collection status."""
    print("=" * 60)
    print("CONFIGURATION CHECK")
    print("=" * 60)
    
    # Check settings
    print("\n📋 Application Settings:")
    print(f"  EMBED_PROVIDER: {settings.embed_provider}")
    print(f"  LLM_PROVIDER: {settings.llm_provider}")
    
    if settings.embed_provider.lower() == "ollama":
        print(f"  OLLAMA_EMBED_MODEL: {settings.ollama_embed_model}")
        model_name = settings.ollama_embed_model.lower()
        expected_dim = DEFAULT_VECTOR_DIMENSIONS.get(model_name, DEFAULT_VECTOR_DIMENSIONS.get("ollama", 768))
        print(f"  Expected vector dimension: {expected_dim}")
    else:
        print(f"  OPENAI_EMBED_MODEL: {settings.openai_embed_model}")
        model_name = settings.openai_embed_model.lower()
        expected_dim = DEFAULT_VECTOR_DIMENSIONS.get(model_name, 1536)
        print(f"  Expected vector dimension: {expected_dim}")
    
    print(f"\n  QDRANT_URL: {settings.qdrant_url}")
    print(f"  QDRANT_COLLECTION: {settings.qdrant_collection}")
    
    # Check Qdrant collection
    print("\n🔍 Qdrant Collection Status:")
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collection_name = settings.qdrant_collection
        
        try:
            collection_info = client.get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size
            points_count = collection_info.points_count
            
            print(f"  Collection exists: ✓")
            print(f"  Current vector dimension: {existing_dim}")
            print(f"  Points in collection: {points_count}")
            
            if existing_dim != expected_dim:
                print(f"\n  ⚠️  MISMATCH DETECTED!")
                print(f"     Expected: {expected_dim} (from model config)")
                print(f"     Current:  {existing_dim} (in collection)")
                print(f"\n  💡 Solution:")
                if points_count == 0:
                    print(f"     Collection is empty. It will be auto-deleted and recreated on next server restart.")
                else:
                    print(f"     Collection has {points_count} points.")
                    print(f"     You need to delete it manually:")
                    print(f"     python delete_collection.py")
            else:
                print(f"\n  ✓ Dimension matches! ({existing_dim})")
                
        except Exception as e:
            error_str = str(e).lower()
            if "doesn't exist" in error_str or "not found" in error_str:
                print(f"  Collection does not exist")
                print(f"  It will be created with dimension {expected_dim} on first use")
            else:
                print(f"  Error checking collection: {e}")
                
    except Exception as e:
        print(f"  ❌ Error connecting to Qdrant: {e}")
        print(f"     Make sure Qdrant is running at {settings.qdrant_url}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_config()

