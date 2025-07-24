# 🔍 Hybrid Charity Client Finder

A comprehensive AI-powered system that identifies whether a charity is a client using multiple search strategies: vector database search, CCN matching, and fuzzy name lookup.

## 🚀 Features

- **🔍 Vector Database Search**: Semantic similarity search using AI embeddings via Qdrant
- **🎯 CCN Matching**: Direct matching of charity commission numbers with client database
- **📋 Fuzzy Lookup**: Name matching in client lookup table (OrgName & OrgName_Sub)
- **🎛️ Hybrid Results**: Combines all search methods with intelligent ranking
- **✅ Clear Client Decision**: Prominent Y/N decision with detailed match information
- **📊 Interactive UI**: Beautiful Streamlit interface with comprehensive result display

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Search Engine  │────│  Qdrant Vector │
│                 │    │                 │    │    Database     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                       ┌─────────────────┐
                       │  Client Lookup  │
                       │  Table (Excel)  │
                       └─────────────────┘
```

## 📁 Project Structure

```
charity-client-finder/
├── charity_search_app.py          # Main Streamlit application
├── src/
│   └── charity_search_engine.py   # Core search engine logic
├── data/
│   ├── backfill_data_ccn/         # Client lookup data
│   │   └── ClientCCNs.xlsx        # Client database
│   └── charity_commission_data/    # Charity Commission data
│       ├── publicextract.charity.json
│       ├── publicextract.charity_classification.json
│       ├── publicextract.charity_other_names.json
│       └── publicextract.charity_trustee.json
├── config/
├── docs/
├── requirements.txt               # Python dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Multi-container setup
└── README.md                    # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- Qdrant vector database (local or cloud)

### Local Development

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd charity-client-finder
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Start Qdrant (if running locally)**:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

5. **Run the application**:
```bash
streamlit run charity_search_app.py
```

6. **Access the app**: Open http://localhost:8501

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Option 2: Docker Build & Run
```bash
docker build -t charity-client-finder .
docker run -p 8501:8501 charity-client-finder
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy automatically

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Google Cloud Run
```bash
gcloud run deploy --source . --platform managed
```

## 📊 Usage

### Basic Search
1. Enter charity name (e.g., "Barnardo's", "RSPCA", "Action Aid")
2. Click "🔍 Search"
3. View results across multiple tabs:
   - **Summary**: Overview of matches found
   - **Vector Results**: Semantic search results
   - **Lookup Results**: Fuzzy name matches
   - **All Matches**: Combined hybrid results

### Search Methods
- **Vector Search**: Finds semantically similar charities
- **CCN Matching**: Matches charity commission numbers
- **Fuzzy Lookup**: Name-based matching with configurable thresholds

### Result Prioritization
1. **CCN Matches** (Highest priority)
2. **Fuzzy Lookup Matches** (Medium priority)  
3. **Vector-only Results** (Display purposes)

## 🔧 Configuration

### Environment Variables
```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
DATABASE_URL=postgresql://...
```

### Search Parameters
- **Max Results**: 5-50 results
- **Score Threshold**: 0.1-1.0 similarity threshold
- **Fuzzy Threshold**: 70% default for name matching

## 📈 Performance

- **Vector Search**: ~0.1-0.5 seconds
- **Fuzzy Lookup**: ~0.05-0.2 seconds
- **Total Search Time**: ~0.3-1.0 seconds
- **Registered Charities Only**: Filters for active organizations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the search logs for debugging

## 🔮 Future Enhancements

- [ ] User authentication system
- [ ] API endpoint for programmatic access
- [ ] Advanced analytics dashboard
- [ ] Machine learning model improvements
- [ ] Real-time data synchronization
- [ ] Multi-language support

---

**Built with ❤️ using Streamlit, Qdrant, and AI embeddings** 