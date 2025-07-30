#!/usr/bin/env python3
"""
Setup script for TextRank + SimpleGNN Streamlit App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for Streamlit app"""
    print("🚀 Setting up TextRank + SimpleGNN Streamlit App")
    print("=" * 50)
    
    # Install Streamlit requirements
    print("📦 Installing Streamlit requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "streamlit_requirements.txt"])
        print("✅ Streamlit requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    
    # Download NLTK data
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
    
    return True

def create_streamlit_config():
    """Create Streamlit configuration file"""
    config_content = """
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
"""
    
    try:
        with open('.streamlit/config.toml', 'w') as f:
            f.write(config_content)
        print("✅ Streamlit configuration created!")
    except Exception as e:
        print(f"⚠️  Warning: Could not create Streamlit config: {e}")

def main():
    """Main setup function"""
    # Create .streamlit directory
    os.makedirs('.streamlit', exist_ok=True)
    
    # Install requirements
    if install_requirements():
        # Create configuration
        create_streamlit_config()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 To run the Streamlit app:")
        print("   streamlit run streamlit_app.py")
        print("\n🌐 The app will be available at: http://localhost:8501")
        print("\n📁 Files created:")
        print("   - streamlit_app.py (Main app)")
        print("   - streamlit_requirements.txt (Dependencies)")
        print("   - .streamlit/config.toml (Configuration)")
        print("   - streamlit_setup.py (This setup script)")
    else:
        print("❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 