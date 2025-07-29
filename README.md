# Charity Client Finder

A sophisticated search application that helps match potential clients with UK charities using advanced semantic search and fuzzy matching algorithms.

## 🚀 Features

- **Hybrid Search**: Combines vector-based semantic search with fuzzy string matching
- **Smart Abbreviation Handling**: Automatically expands common abbreviations (e.g., "RSPCA" → "Royal Society for the Prevention of Cruelty to Animals")
- **Multiple Match Types**: Provides both charity commission data and client lookup results
- **Configurable Thresholds**: Adjustable similarity thresholds for precise matching
- **Real-time Results**: Fast search across 390k+ charity records

## 🛠️ Development Workflow

### Quick Start Commands

```bash
# Start local development
./dev-workflow.sh start-local

# Check environment status  
./dev-workflow.sh status

# Deploy to production
./dev-workflow.sh deploy

# Stop local server
./dev-workflow.sh stop-local
```

### 🔄 Step-by-Step Development Process

1. **Start Local Development:**
   ```bash
   ./dev-workflow.sh start-local
   ```
   - Starts app at `http://localhost:8501`
   - Connects to local Qdrant at `localhost:6333`
   - Uses your local data (392k+ charity records)

2. **Make Your Changes:**
   - Edit code in your favorite editor
   - Changes are automatically reflected (Streamlit auto-reloads)

3. **Test Locally:**
   - Visit `http://localhost:8501`
   - Test your changes with real data
   - Try searches like "rspca", "barnardo's", etc.

4. **Deploy When Ready:**
   ```bash
   ./dev-workflow.sh deploy
   ```
   - Commits your changes to Git
   - Pushes to GitHub
   - Streamlit Cloud auto-deploys in ~2 minutes

### 🌍 Environment Differences

| Environment | Code Source | Database | URL |
|------------|-------------|----------|-----|
| **Local** | Your files | `localhost:6333` | `http://localhost:8501` |
| **Production** | GitHub repo | Qdrant Cloud | Streamlit Cloud URL |

## 🔍 Search Features

### Enhanced Fuzzy Matching
- **Partial matching**: Finds "RSPCA - Bradford Branch" when searching "rspca"
- **Token-based matching**: Handles multi-word queries intelligently  
- **Abbreviation expansion**: Recognizes 50+ common charity abbreviations
- **Adaptive thresholds**: Lower thresholds for short queries

### Search Algorithms
1. **Vector Search**: Semantic similarity using sentence transformers
2. **Fuzzy Lookup**: String similarity with multiple algorithms:
   - Partial ratio matching
   - Token sort ratio
   - Token set ratio  
   - Containment matching

## 📊 Data Sources

- **Charity Commission Data**: 392k+ UK registered charities
- **Client Lookup Table**: Curated client organization mappings
- **Classification Data**: Charity categories and purposes
- **Trustee Information**: Charity leadership data

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Search Engine**: Custom hybrid search with Qdrant vector database
- **NLP**: Sentence Transformers (all-MiniLM-L6-v2)
- **Fuzzy Matching**: FuzzyWuzzy library
- **Data Processing**: Pandas
- **Deployment**: Streamlit Cloud + Qdrant Cloud

## 📝 Configuration

### Local Development
- No configuration needed
- Uses local Qdrant instance automatically
- Loads data from local JSON files

### Production Deployment  
- Automatically configured via Streamlit Cloud secrets
- Uses Qdrant Cloud for vector storage
- Environment variables handled by deployment platform

## 🧪 Testing

### Search Examples to Try:

**Abbreviations:**
- "rspca" → Royal Society for the Prevention of Cruelty to Animals branches
- "nspcc" → National Society for the Prevention of Cruelty to Children  
- "brc" → British Red Cross

**Partial Names:**
- "barnardo" → Barnardo's and related organizations
- "oxfam" → Oxfam and international development charities
- "cancer research" → Various cancer research organizations

**Threshold Testing:**
- High threshold (0.8+): Exact matches only
- Medium threshold (0.5-0.7): Good balance
- Low threshold (0.3-0.4): Broader matching, good for abbreviations

## 📈 Performance

- **Search Speed**: Sub-second response times
- **Data Volume**: 390k+ charity records indexed
- **Memory Usage**: Optimized for cloud deployment
- **Scalability**: Horizontal scaling via Qdrant Cloud

## 🚀 Deployment

The application automatically deploys to Streamlit Cloud when you push changes to the main branch:

1. Make changes locally
2. Test at `http://localhost:8501`  
3. Run `./dev-workflow.sh deploy`
4. Production updates in ~2 minutes

## 📞 Support

For questions or issues with the development workflow, check:
- Local logs in terminal when running `./dev-workflow.sh start-local`
- Streamlit Cloud deployment logs for production issues
- GitHub Actions for deployment status # Force deployment update
# CRITICAL NSPCC FIX - Force deployment
# CRITICAL QDRANT ERROR HANDLING FIX
# FRESH DEPLOYMENT: Ensure perfect local/production parity
# CRITICAL FIX: Prevent Pooling.init() error and improve error handling
# CRITICAL: Ensure local and deployed apps use same ClientCCNs file
# CRITICAL: Production app now uses Excel file instead of fallback table
# DEBUG: Enhanced logging for production troubleshooting
