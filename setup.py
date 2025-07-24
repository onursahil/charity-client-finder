#!/usr/bin/env python3
"""
Setup script for Charity Commission Search Engine
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def check_qdrant_installation():
    """Check if Qdrant is available and install if needed."""
    print("🔍 Checking Qdrant installation...")
    
    # Try to import qdrant-client
    try:
        import qdrant_client
        print("✅ Qdrant client library is installed")
    except ImportError:
        print("❌ Qdrant client library not found")
        return False
    
    return True

def start_qdrant_docker():
    """Start Qdrant using Docker."""
    print("🐳 Starting Qdrant with Docker...")
    
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not installed or not available")
        print("Please install Docker from https://docs.docker.com/get-docker/")
        return False
    
    # Check if Qdrant container is already running
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        if "qdrant" in result.stdout:
            print("✅ Qdrant is already running")
            return True
    except subprocess.CalledProcessError:
        pass
    
    # Start Qdrant container
    try:
        subprocess.run([
            "docker", "run", "-d",
            "--name", "qdrant",
            "-p", "6333:6333",
            "-p", "6334:6334",
            "qdrant/qdrant"
        ], check=True)
        print("✅ Qdrant container started successfully")
        
        # Wait for Qdrant to be ready
        print("⏳ Waiting for Qdrant to be ready...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:6333/collections", timeout=5)
                if response.status_code == 200:
                    print("✅ Qdrant is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        
        print("❌ Qdrant failed to start within 30 seconds")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Qdrant: {e}")
        return False

def check_data_files():
    """Check if charity commission data files exist."""
    print("📁 Checking data files...")
    
    data_dir = Path("data/charity_commission_data")
    required_files = [
        "publicextract.charity.json",
        "publicextract.charity_classification.json",
        "publicextract.charity_other_names.json",
        "publicextract.charity_trustee.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        print("Please ensure all charity commission data files are in the data/charity_commission_data directory")
        return False
    
    return True

def run_initial_indexing():
    """Run initial indexing of the charity data."""
    print("🔍 Running initial indexing...")
    try:
        from charity_search_engine import CharitySearchEngine
        
        # Initialize search engine
        search_engine = CharitySearchEngine()
        
        # Load and index data
        data = search_engine.load_charity_data()
        documents = search_engine.prepare_search_documents(data)
        search_engine.index_documents(documents)
        
        # Get collection info
        info = search_engine.get_collection_info()
        print(f"✅ Indexing completed! {info.get('points_count', 0):,} documents indexed")
        
    except Exception as e:
        print(f"❌ Failed to run initial indexing: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🏛️ Charity Commission Search Engine Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Check Qdrant installation
    if not check_qdrant_installation():
        print("❌ Qdrant client library installation failed")
        sys.exit(1)
    
    # Start Qdrant
    if not start_qdrant_docker():
        print("❌ Failed to start Qdrant")
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("❌ Data files check failed")
        sys.exit(1)
    
    # Run initial indexing
    print("\n🚀 Starting initial data indexing...")
    if not run_initial_indexing():
        print("❌ Initial indexing failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎉 You can now run the search engine:")
    print("   Streamlit app: streamlit run charity_search_app.py")
    print("   Command line: python charity_search_engine.py")
    print("\n📖 For more information, check the README.md file")

if __name__ == "__main__":
    main() 