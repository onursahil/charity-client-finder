import streamlit as st
import pandas as pd
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from charity_search_engine import CharitySearchEngine
import json
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Hybrid Charity Client Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .client-decision-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .client-yes {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .client-no {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .search-method-header {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .match-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ccn-match {
        border-left: 4px solid #28a745;
    }
    .fuzzy-match {
        border-left: 4px solid #ffc107;
    }
    .vector-match {
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .match-type-badge {
        display: inline-block;
        background-color: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .connection-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .connected {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .disconnected {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_search_engine():
    """Initialize the search engine with caching."""
    try:
        return CharitySearchEngine()
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        
        # Check if we're in cloud deployment
        if os.getenv('QDRANT_URL') or os.getenv('QDRANT_API_KEY'):
            st.info("Please make sure your Qdrant Cloud credentials are correctly configured.")
            st.markdown("""
            **Required Environment Variables:**
            - `QDRANT_URL`: Your Qdrant Cloud cluster URL
            - `QDRANT_API_KEY`: Your Qdrant Cloud API key
            """)
        else:
            st.info("Please make sure Qdrant is running on localhost:6333")
        return None

def check_connection_status(search_engine):
    """Check and display Qdrant connection status."""
    if search_engine is None:
        st.markdown("""
        <div class="connection-status disconnected">
            ❌ QDRANT DISCONNECTED: Unable to connect to vector database
        </div>
        """, unsafe_allow_html=True)
        return False
    
    try:
        # Test connection by getting collection info
        info = search_engine.get_collection_info()
        if info and info.get('points_count', 0) > 0:
            st.markdown(f"""
            <div class="connection-status connected">
                ✅ QDRANT CONNECTED: {info.get('points_count', 0):,} charity records indexed
            </div>
            """, unsafe_allow_html=True)
            return True
        else:
            st.markdown("""
            <div class="connection-status disconnected">
                ⚠️ QDRANT CONNECTED BUT NO DATA: Database is empty - please load data first
            </div>
            """, unsafe_allow_html=True)
            return False
    except Exception as e:
        st.markdown(f"""
        <div class="connection-status disconnected">
            ❌ QDRANT ERROR: {str(e)}
        </div>
        """, unsafe_allow_html=True)
        return False

def load_and_index_data(search_engine):
    """Load and index charity data into Qdrant."""
    if search_engine is None:
        st.error("Search engine not initialized")
        return False
    
    try:
        # Load data
        with st.spinner("Loading charity commission data..."):
            data = search_engine.load_charity_data()
            
        if not data or data.get('charities', pd.DataFrame()).empty:
            st.error("No charity data found. Please check data files.")
            return False
            
        st.success(f"Loaded {len(data['charities'])} charity records")
        
        # Prepare documents
        with st.spinner("Preparing documents for indexing..."):
            documents = search_engine.prepare_search_documents(data)
            
        if not documents:
            st.error("No documents to index")
            return False
            
        st.success(f"Prepared {len(documents)} documents for indexing")
        
        # Index documents
        with st.spinner("Indexing documents into Qdrant... This may take several minutes."):
            search_engine.index_documents(documents)
            
        st.success("✅ Data loading and indexing completed successfully!")
        
        # Refresh the cached search engine
        st.cache_resource.clear()
        
        return True
        
    except Exception as e:
        st.error(f"Error during data loading: {e}")
        return False

def display_client_decision(is_client: bool, client_decision: str):
    """Display the prominent client decision box."""
    if is_client:
        st.markdown(f"""
        <div class="client-decision-box client-yes">
            🎯 CLIENT FOUND: {client_decision}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="client-decision-box client-no">
            ❌ NOT A CLIENT: {client_decision}
        </div>
        """, unsafe_allow_html=True)

def display_best_match(best_match: Dict[str, Any]):
    """Display the best match details."""
    if not best_match:
        return
        
    st.subheader("🥇 Best Match Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Charity Name:** {best_match.get('charity_name', 'N/A')}")
        st.write(f"**CCN:** {best_match.get('registered_charity_number', 'N/A')}")
        st.write(f"**Match Type:** {best_match.get('match_type', 'N/A')}")
        
    with col2:
        st.write(f"**Vector Score:** {best_match.get('vector_score', 0):.3f}")
        st.write(f"**Name Similarity:** {best_match.get('name_similarity', 0):.1f}%")
        st.write(f"**Combined Score:** {best_match.get('combined_score', 0):.3f}")

def display_vector_results(vector_results: list):
    """Display vector search results."""
    if not vector_results:
        st.info("No vector search results found")
        return
        
    st.markdown('<div class="search-method-header">🔍 Vector Database Search Results</div>', unsafe_allow_html=True)
    
    for i, result in enumerate(vector_results[:10], 1):  # Limit to top 10
        charity_name = result.get('charity_name', 'Unknown')
        with st.expander(f"Vector Result #{i}: {charity_name} (Score: {result.get('score', 0):.3f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Charity Name:** {result.get('charity_name', 'N/A')}")
                st.write(f"**CCN:** {result.get('registered_charity_number', 'N/A')}")
                st.write(f"**Status:** {result.get('charity_registration_status', 'N/A')}")
                st.write(f"**Vector Score:** {result.get('score', 0):.3f}")
                
            with col2:
                st.write(f"**Address:** {result.get('address', 'N/A')}")
                st.write(f"**Latest Income:** {result.get('latest_income', 'N/A')}")
                st.write(f"**Latest Expenditure:** {result.get('latest_expenditure', 'N/A')}")
            
            if result.get('charity_activities'):
                st.write(f"**Activities:** {result.get('charity_activities')}")

def display_lookup_results(lookup_results: list):
    """Display fuzzy lookup search results."""
    if not lookup_results:
        st.info("No fuzzy lookup results found")
        return
        
    st.markdown('<div class="search-method-header">📋 Fuzzy Lookup Table Results</div>', unsafe_allow_html=True)
    
    # Performance optimization: Limit displayed results to prevent UI slowdown
    MAX_DISPLAY_RESULTS = 20
    total_results = len(lookup_results)
    
    if total_results > MAX_DISPLAY_RESULTS:
        st.warning(f"⚠️ Showing top {MAX_DISPLAY_RESULTS} results out of {total_results} total matches for performance")
        display_results = lookup_results[:MAX_DISPLAY_RESULTS]
    else:
        display_results = lookup_results
    
    for i, result in enumerate(display_results, 1):
        match_field = result.get('match_field', 'Unknown')
        similarity = result.get('name_similarity', 0)
        
        # Get the actual matched organization name
        matched_name = result.get('charity_name', 'Unknown')
        if result.get('client_org_name'):
            matched_name = result.get('client_org_name')
            if result.get('client_org_sub') and result.get('client_org_sub') != result.get('client_org_name'):
                matched_name += f" / {result.get('client_org_sub')}"
        
        with st.expander(f"Lookup Result #{i}: {matched_name} ({match_field}, {similarity}% match)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Matched Name:** {result.get('charity_name', 'N/A')}")
                st.write(f"**Match Field:** {match_field}")
                st.write(f"**Client CCN:** {result.get('client_ccn', 'N/A')}")
                st.write(f"**Name Similarity:** {similarity}%")
                
            with col2:
                st.write(f"**Client Org Name:** {result.get('client_org_name', 'N/A')}")
                st.write(f"**Client Org Sub:** {result.get('client_org_sub', 'N/A')}")
                st.write(f"**Search Score:** {result.get('score', 0):.3f}")
    
    # Show summary if results were limited
    if total_results > MAX_DISPLAY_RESULTS:
        st.info(f"💡 To see all {total_results} results, try refining your search query or increasing the score threshold")

def display_all_matches(all_matches: list):
    """Display all hybrid matches ranked by priority."""
    if not all_matches:
        st.info("No matches found across all search methods")
        return
        
    st.markdown('<div class="search-method-header">🎯 All Hybrid Matches (Ranked by Priority)</div>', unsafe_allow_html=True)
    
    # Performance optimization: Limit displayed results to prevent UI slowdown
    MAX_DISPLAY_RESULTS = 30
    total_results = len(all_matches)
    
    if total_results > MAX_DISPLAY_RESULTS:
        st.warning(f"⚠️ Showing top {MAX_DISPLAY_RESULTS} results out of {total_results} total matches for performance")
        display_matches = all_matches[:MAX_DISPLAY_RESULTS]
    else:
        display_matches = all_matches
    
    for i, match in enumerate(display_matches, 1):
        source = match.get('source', 'Unknown')
        match_type = match.get('match_type', 'Unknown')
        is_client = match.get('is_client', False)
        
        # Determine card style based on source
        card_class = "match-card "
        if source == 'ccn_match':
            card_class += "ccn-match"
            icon = "🎯"
        elif 'fuzzy_lookup' in source:
            card_class += "fuzzy-match"
            icon = "📋"
        else:
            card_class += "vector-match"
            icon = "🔍"
            
        client_status = "✅ CLIENT" if is_client else "❌ NOT CLIENT"
        
        # Get the actual matched organization name
        display_name = match.get('charity_name', 'Unknown')
        matched_org_name = None
        
        # For fuzzy lookup results, show the actual matched organization name
        if 'fuzzy_lookup' in match.get('source', ''):
            details = match.get('details', {})
            if details.get('client_org_name'):
                matched_org_name = details.get('client_org_name')
                if details.get('client_org_sub') and details.get('client_org_sub') != details.get('client_org_name'):
                    matched_org_name += f" / {details.get('client_org_sub')}"
            display_name = matched_org_name or display_name
        
        # For vector results, show the actual charity name from the database  
        elif match.get('source') == 'vector_search':
            details = match.get('details', {})
            if details.get('charity_name'):
                display_name = details.get('charity_name')
        
        # For CCN matches, show the vector result charity name
        elif match.get('source') == 'ccn_match':
            details = match.get('details', {})
            if details.get('charity_name'):
                display_name = details.get('charity_name')
        
        with st.expander(f"{icon} Match #{i}: {display_name} - {client_status}"):
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Source:** {source}")
                st.write(f"**Match Type:** {match_type}")
                st.write(f"**Is Client:** {'Yes' if is_client else 'No'}")
                
                # Show matched organization details for lookup results
                if 'fuzzy_lookup' in match.get('source', ''):
                    details = match.get('details', {})
                    if details.get('client_org_name'):
                        st.write(f"**Matched Org:** {details.get('client_org_name')}")
                    if details.get('client_org_sub') and details.get('client_org_sub') != details.get('client_org_name'):
                        st.write(f"**Org Sub-name:** {details.get('client_org_sub')}")
                
            with col2:
                st.write(f"**Vector Score:** {match.get('vector_score', 0):.3f}")
                st.write(f"**Name Similarity:** {match.get('name_similarity', 0):.1f}%")
                st.write(f"**Combined Score:** {match.get('combined_score', 0):.3f}")
                
            with col3:
                st.write(f"**Priority:** {match.get('priority', 'N/A')}")
                st.write(f"**CCN Match:** {'Yes' if match.get('ccn_match') else 'No'}")
                st.write(f"**CCN:** {match.get('registered_charity_number', 'N/A')}")
                
                # Show which field was matched for lookup results
                if 'fuzzy_lookup' in match.get('source', ''):
                    details = match.get('details', {})
                    if details.get('match_field'):
                        st.write(f"**Match Field:** {details.get('match_field')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show summary if results were limited
    if total_results > MAX_DISPLAY_RESULTS:
        st.info(f"💡 To see all {total_results} results, try refining your search query or increasing the score threshold")

def main():
    # Initialize search engine
    search_engine = initialize_search_engine()
    
    # Main title
    st.markdown('<h1 class="main-header">🔍 Hybrid Charity Client Finder</h1>', unsafe_allow_html=True)
    
    # Check connection status
    is_connected = check_connection_status(search_engine)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Data Management", "ℹ️ Info"])
    
    with tab3:
        st.subheader("About This Tool")
        st.markdown("""
        This tool combines multiple search methods to identify whether a charity is a client:
        - **🔍 Vector Database Search**: Semantic similarity search using AI embeddings
        - **🎯 CCN Matching**: Direct matching of charity commission numbers
        - **📋 Fuzzy Lookup**: Name matching in client lookup table (OrgName & OrgName_Sub)
        
        **Deployment Info:**
        """)
        
        # Show environment info
        qdrant_url = os.getenv('QDRANT_URL')
        if qdrant_url:
            st.write("🌐 **Environment**: Cloud (Streamlit Cloud + Qdrant Cloud)")
            st.write(f"🔗 **Qdrant URL**: `{qdrant_url[:50]}...`")
        elif os.getenv('QDRANT_API_KEY'):
            st.write("🌐 **Environment**: Cloud (Custom + Qdrant Cloud)")
        else:
            st.write("🏠 **Environment**: Local Development")
    
    with tab2:
        st.subheader("📊 Data Management")
        
        if search_engine is None:
            st.error("Cannot manage data - search engine not initialized")
            return
            
        # Show collection info
        try:
            info = search_engine.get_collection_info()
            if info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Collection", info.get('name', 'N/A'))
                with col2:
                    st.metric("Records", f"{info.get('points_count', 0):,}")
                with col3:
                    st.metric("Status", info.get('status', 'N/A'))
            else:
                st.warning("No collection information available")
        except Exception as e:
            st.error(f"Error getting collection info: {e}")
        
        st.markdown("---")
        
        # Debug section for client lookup table
        st.subheader("🔍 Debug: Client Lookup Table")
        
        if hasattr(search_engine, 'client_lookup') and search_engine.client_lookup is not None:
            st.success(f"✅ Client lookup table loaded: {len(search_engine.client_lookup)} records")
            st.write(f"**Columns:** {list(search_engine.client_lookup.columns)}")
            
            # Show RSPCA entries
            rspca_entries = search_engine.client_lookup[
                search_engine.client_lookup['OrgName'].str.contains('rspca|royal society for the prevention of cruelty to animals', case=False, na=False)
            ]
            st.write(f"**RSPCA entries found:** {len(rspca_entries)}")
            
            if len(rspca_entries) > 0:
                st.write("**Sample RSPCA entries:**")
                for i, (_, entry) in enumerate(rspca_entries.head(3).iterrows()):
                    st.write(f"  {i+1}. CCN: {entry['ccn']}, OrgName: {entry['OrgName']}")
            else:
                st.error("❌ No RSPCA entries found in client lookup table!")
                
            # Test fuzzy search directly
            if st.button("🧪 Test 'r.s.p.c.a' Search"):
                try:
                    results = search_engine._fuzzy_lookup_search("r.s.p.c.a", threshold=50)
                    st.write(f"**Fuzzy search results for 'r.s.p.c.a': {len(results)} matches**")
                    for i, result in enumerate(results[:5]):
                        st.write(f"  {i+1}. {result['charity_name']} (Score: {result['score']:.3f})")
                except Exception as e:
                    st.error(f"Error testing fuzzy search: {e}")
        else:
            st.error("❌ Client lookup table not loaded!")
        
        st.markdown("---")
        
        # Data loading section
        st.subheader("🔄 Load Data")
        
        # File upload option
        st.info("""
        **Option 1: Upload Data Files** - Upload your JSON files directly here.
        """)
        
        with st.expander("📁 Upload Data Files", expanded=False):
            uploaded_files = {}
            file_names = [
                ('charity', 'publicextract.charity.json'),
                ('classification', 'publicextract.charity_classification.json'), 
                ('other_names', 'publicextract.charity_other_names.json'),
                ('trustee', 'publicextract.charity_trustee.json')
            ]
            
            for key, filename in file_names:
                uploaded_file = st.file_uploader(
                    f"Upload {filename}",
                    type=['json'],
                    key=f"upload_{key}",
                    help=f"Upload the {filename} file"
                )
                if uploaded_file:
                    uploaded_files[key] = uploaded_file
            
            if len(uploaded_files) == 4:
                if st.button("🚀 Process Uploaded Files", type="primary"):
                    try:
                        # Create temporary directory structure
                        import tempfile
                        import shutil
                        
                        temp_dir = tempfile.mkdtemp()
                        data_dir = os.path.join(temp_dir, "charity_commission_data")
                        os.makedirs(data_dir, exist_ok=True)
                        
                        # Save uploaded files
                        for key, uploaded_file in uploaded_files.items():
                            file_mapping = {
                                'charity': 'publicextract.charity.json',
                                'classification': 'publicextract.charity_classification.json',
                                'other_names': 'publicextract.charity_other_names.json',
                                'trustee': 'publicextract.charity_trustee.json'
                            }
                            
                            file_path = os.path.join(data_dir, file_mapping[key])
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Load and index data
                        with st.spinner("Processing uploaded files..."):
                            data = search_engine.load_charity_data(data_dir)
                            
                        if data and not data.get('charities', pd.DataFrame()).empty:
                            with st.spinner("Preparing documents..."):
                                documents = search_engine.prepare_search_documents(data)
                                
                            if documents:
                                with st.spinner("Indexing documents... This may take several minutes."):
                                    search_engine.index_documents(documents)
                                    
                                st.success("✅ Files uploaded and indexed successfully!")
                                
                                # Clean up
                                shutil.rmtree(temp_dir)
                                st.cache_resource.clear()
                                st.experimental_rerun()
                            else:
                                st.error("No documents to index")
                        else:
                            st.error("No valid charity data found in uploaded files")
                            
                        # Clean up on error
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                            
                    except Exception as e:
                        st.error(f"Error processing uploaded files: {e}")
                        if 'temp_dir' in locals() and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
            else:
                st.info(f"Please upload all 4 files. Currently uploaded: {len(uploaded_files)}/4")
        
        st.markdown("---")
        
        # Cloud/Local data loading
        st.info("""
        **Option 2: Load from Cloud Storage** - If you've set up cloud storage URLs in secrets.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Load and Index Data", type="primary"):
                if load_and_index_data(search_engine):
                    st.experimental_rerun()
        
        with col2:
            if st.button("🗑️ Delete Collection", help="⚠️ This will delete all indexed data"):
                if st.session_state.get('confirm_delete', False):
                    try:
                        search_engine.delete_collection()
                        st.success("Collection deleted successfully")
                        st.cache_resource.clear()
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting collection: {e}")
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click again to confirm deletion")
    
    with tab1:
        if not is_connected:
            st.error("⚠️ Cannot search - database not connected or empty. Please check the Data Management tab.")
            return
    
    # Search interface
    with st.container():
        st.subheader("Search for a Charity")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter charity name or keywords:",
                placeholder="e.g., Barnardo's, British Red Cross... (Press Enter to search)",
                help="Enter the name of the charity you want to search for and press Enter or click Search",
                key="search_input"
            )
            
        with col2:
            st.write("")  # Spacing
            search_button = st.button("🔍 Search", type="primary")
    
    # Search parameters in sidebar
    with st.sidebar:
        st.header("Search Parameters")
        
        limit = st.slider("Max Results", min_value=5, max_value=50, value=10, step=5)
        score_threshold = st.slider("Score Threshold", min_value=0.1, max_value=1.0, value=0.60, step=0.1)
        
        st.header("Search Options")
        show_vector_details = st.checkbox("Show Vector Results Details", value=True)
        show_lookup_details = st.checkbox("Show Lookup Results Details", value=True)
        show_all_matches = st.checkbox("Show All Hybrid Matches", value=True)
    
    # Perform search (trigger on button click or when query is entered)
    should_search = (search_button and query) or (query and len(query.strip()) > 2)
    
    if should_search:
        with st.spinner("🔍 Performing hybrid search..."):
            start_time = time.time()
            
            # Perform hybrid search
            results = search_engine.hybrid_client_search(
                query=query,
                limit=limit,
                score_threshold=score_threshold
            )
            
            search_time = time.time() - start_time
            
        # Display results
        st.success(f"Search completed in {search_time:.2f} seconds")
        
        # Display client decision prominently
        display_client_decision(results['is_client'], results['client_decision'])
        
        # Display best match if found
        if results['best_match']:
            display_best_match(results['best_match'])
        
        # Create tabs for different result types
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📊 Summary ({len(results.get('all_matches', []))} matches)",
            f"🔍 Vector Results ({len(results.get('vector_results', []))})",
            f"📋 Lookup Results ({len(results.get('lookup_results', []))})",
            "🎯 All Matches"
        ])
        
        with tab1:
            st.subheader("Search Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Vector Results", len(results.get('vector_results', [])))
            with col2:
                st.metric("Lookup Results", len(results.get('lookup_results', [])))
            with col3:
                st.metric("Total Matches", len(results.get('all_matches', [])))
            with col4:
                st.metric("Client Status", "YES" if results['is_client'] else "NO")
            
            # Display search query info
            st.write(f"**Search Query:** {query}")
            st.write(f"**Search Time:** {search_time:.2f} seconds")
            
        with tab2:
            if show_vector_details:
                display_vector_results(results.get('vector_results', []))
            else:
                st.info(f"Found {len(results.get('vector_results', []))} vector search results. Enable 'Show Vector Results Details' in sidebar to view.")
                
        with tab3:
            if show_lookup_details:
                display_lookup_results(results.get('lookup_results', []))
            else:
                st.info(f"Found {len(results.get('lookup_results', []))} lookup results. Enable 'Show Lookup Results Details' in sidebar to view.")
                
        with tab4:
            if show_all_matches:
                display_all_matches(results.get('all_matches', []))
            else:
                st.info(f"Found {len(results.get('all_matches', []))} total matches. Enable 'Show All Hybrid Matches' in sidebar to view.")
    
    elif query and len(query.strip()) <= 2:
        st.info("Type at least 3 characters to search automatically, or click the Search button")
    
    # Footer
    st.markdown("---")
    st.markdown("**Hybrid Client Finder** - Combining vector search, CCN matching, and fuzzy lookup for comprehensive charity client identification")

if __name__ == "__main__":
    main() 