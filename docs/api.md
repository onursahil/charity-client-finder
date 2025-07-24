# API Documentation

## CharitySearchEngine Class

### Overview
The `CharitySearchEngine` class provides hybrid search capabilities for identifying charity clients using vector search, CCN matching, and fuzzy lookup.

### Initialization

```python
from src.charity_search_engine import CharitySearchEngine

engine = CharitySearchEngine(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="charity_commission"
)
```

### Main Search Method

#### `hybrid_client_search(query, limit=10, score_threshold=0.3, filters=None)`

Performs comprehensive hybrid search combining all search methods.

**Parameters:**
- `query` (str): Charity name or keywords to search for
- `limit` (int): Maximum number of vector search results (default: 10)
- `score_threshold` (float): Minimum similarity score for vector search (default: 0.3)
- `filters` (dict, optional): Additional filters for vector search

**Returns:**
```python
{
    'is_client': bool,              # True if charity is a client
    'client_decision': str,         # 'Y' or 'N'
    'vector_results': list,         # Vector search results
    'lookup_results': list,         # Fuzzy lookup results
    'best_match': dict,             # Best match details
    'all_matches': list,            # All matches ranked by priority
    'query': str                    # Original search query
}
```

**Example:**
```python
results = engine.hybrid_client_search("Barnardo's")
print(f"Is client: {results['client_decision']}")
print(f"Best match: {results['best_match']['charity_name']}")
```

### Search Components

#### Vector Search
- Uses AI embeddings for semantic similarity
- Filters for registered charities only
- Returns top matches with similarity scores

#### CCN Matching
- Matches charity commission numbers between vector results and client database
- Most reliable identification method
- Validates with name similarity

#### Fuzzy Lookup
- Direct search in client lookup table
- Matches against OrgName and OrgName_Sub fields
- Uses 70% similarity threshold by default

### Result Structure

#### Vector Results
```python
{
    'source': 'vector_search',
    'id': int,
    'score': float,
    'charity_name': str,
    'registered_charity_number': str,
    'charity_registration_status': str,
    'charity_activities': str,
    'address': str,
    'latest_income': float,
    'latest_expenditure': float,
    'metadata': dict,
    'text_content': str
}
```

#### Lookup Results
```python
{
    'source': 'fuzzy_lookup_orgname|fuzzy_lookup_orgsub',
    'score': float,
    'charity_name': str,
    'registered_charity_number': str,
    'match_field': str,
    'client_ccn': str,
    'client_org_name': str,
    'client_org_sub': str,
    'name_similarity': int
}
```

#### Best Match
```python
{
    'match_type': str,              # Type of match found
    'charity_name': str,            # Matched charity name
    'registered_charity_number': str,
    'vector_score': float,          # Vector similarity score
    'name_similarity': float,       # Name matching score
    'ccn_match': bool,              # Whether CCN matched
    'combined_score': float,        # Overall match quality
    'priority': int                 # Match priority (1=highest)
}
```

### Utility Methods

#### `search(query, limit=10, score_threshold=0.3, filters=None)`
Backwards-compatible wrapper for `hybrid_client_search()`.

#### `load_charity_data(data_dir="data/charity_commission_data")`
Loads charity commission data from JSON files.

#### `prepare_search_documents(data)`
Prepares documents for vector indexing.

#### `index_documents(documents, batch_size=100)`
Indexes documents into Qdrant vector database.

#### `get_collection_info()`
Returns information about the Qdrant collection.

### Configuration

#### Environment Variables
- `QDRANT_HOST`: Qdrant server hostname
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_URL`: Full Qdrant URL
- `QDRANT_API_KEY`: API key for Qdrant Cloud

#### Default Settings
- Vector model: `all-MiniLM-L6-v2`
- Vector size: 384 dimensions
- Fuzzy threshold: 70%
- Score threshold: 0.3

### Error Handling

The engine includes comprehensive error handling:
- Connection errors to Qdrant
- Missing data files
- Invalid search parameters
- Embedding generation failures

All errors are logged and gracefully handled to ensure system stability.

### Performance Considerations

- **Vector Search**: ~0.1-0.5 seconds
- **Fuzzy Lookup**: ~0.05-0.2 seconds
- **Total Search**: ~0.3-1.0 seconds
- **Memory Usage**: ~1-2GB depending on data size
- **Concurrent Users**: Supports multiple simultaneous searches

### Best Practices

1. **Indexing**: Index charity data once on startup
2. **Caching**: Consider caching frequent searches
3. **Monitoring**: Monitor search performance and accuracy
4. **Updates**: Regularly update charity data
5. **Scaling**: Use Qdrant Cloud for production scaling 