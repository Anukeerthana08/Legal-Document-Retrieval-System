import streamlit as st
import numpy as np
import faiss
import json
import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import base64
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration if config.py is not available
    INDEX_PATH = "train_faiss_index_legalbert.idx"
    DOC_MAP_PATH = "train_document_map_legalbert.npy"
    CLEANED_FOLDER = "cleaned_texts"
    PDF_FOLDER = "nlp ds"
    NER_OUTPUT_PATH = "legal_ner_output.json"
    EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    NER_MODEL_NAME = "easwar03/legal-bert-base-NER"
    TOP_K = 5
    SIM_CONVERT = lambda d: 1 - d / (1 + d)
    
    LABEL_MAPPING = {
        "LABEL_0": "COURT", "LABEL_1": "PETITIONER", "LABEL_2": "RESPONDENT",
        "LABEL_3": "JUDGE", "LABEL_4": "LAWYER", "LABEL_5": "DATE",
        "LABEL_6": "ORG", "LABEL_7": "GPE", "LABEL_8": "STATUTE",
        "LABEL_9": "PROVISION", "LABEL_10": "PRECEDENT", "LABEL_11": "CASE_NUMBER",
        "LABEL_12": "OTHER_PERSON"
    }
    
    ENTITY_COLORS = {
        "COURT": "#FF6B6B", "PETITIONER": "#4ECDC4", "RESPONDENT": "#45B7D1",
        "JUDGE": "#96CEB4", "LAWYER": "#FFEAA7", "DATE": "#DDA0DD",
        "ORG": "#98D8C8", "GPE": "#F7DC6F", "STATUTE": "#BB8FCE",
        "PROVISION": "#85C1E9", "PRECEDENT": "#F8C471", "CASE_NUMBER": "#82E0AA",
        "OTHER_PERSON": "#F1948A"
    }
    
    QUERY_SUGGESTIONS = [
        "Section 302 IPC murder conviction", "Contract breach damages compensation", 
        "Property dispute boundary rights", "Divorce custody child maintenance",
        "Criminal appeal bail application", "Constitutional Article 14 equality",
        "Service tax penalty waiver", "Land acquisition compensation",
        "Consumer protection deficiency", "Employment termination wrongful"
    ]

class LegalDocumentRetriever:
    def __init__(self):
        self.index = None
        self.document_map = None
        self.embedder = None
        self.ner_data = None
        self.entity_index = {}  # Entity-based document index
        self.document_entities = {}  # Pre-processed document entities
        
    @st.cache_resource
    def load_models_and_data(_self):
        """Load FAISS index, document map, embedder, and NER data"""
        try:
            # Load FAISS index
            if os.path.exists(INDEX_PATH):
                _self.index = faiss.read_index(INDEX_PATH)
                st.success(f"✅ Loaded FAISS index with {_self.index.ntotal} documents")
            else:
                st.error(f"❌ FAISS index not found at {INDEX_PATH}")
                return False
                
            # Load document map
            if os.path.exists(DOC_MAP_PATH):
                _self.document_map = np.load(DOC_MAP_PATH, allow_pickle=True)
                st.success(f"✅ Loaded document map with {len(_self.document_map)} entries")
            else:
                st.error(f"❌ Document map not found at {DOC_MAP_PATH}")
                return False
                
            # Load sentence transformer
            with st.spinner("Loading embedding model..."):
                _self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
                st.success(f"✅ Loaded embedding model: {EMBED_MODEL_NAME}")
                
            # Load NER data
            if os.path.exists(NER_OUTPUT_PATH):
                with open(NER_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    _self.ner_data = json.load(f)
                st.success(f"✅ Loaded NER data with {len(_self.ner_data)} entities")
                
                # Build entity indexes for efficient retrieval
                _self._build_entity_indexes()
                st.success(f"✅ Built entity indexes for {len(_self.entity_index)} entity types")
            else:
                st.warning(f"⚠️ NER data not found at {NER_OUTPUT_PATH}")
                _self.ner_data = []
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading models and data: {str(e)}")
            return False
    
    def _build_entity_indexes(self):
        """Build entity-based indexes for efficient retrieval"""
        self.entity_index = {}
        self.document_entities = {}
        
        confidence_threshold = getattr(globals(), 'ENTITY_CONFIDENCE_THRESHOLD', 0.8)
        
        for entity in self.ner_data:
            if entity['score'] > confidence_threshold:
                file_name = entity['file']
                entity_label = LABEL_MAPPING.get(entity['entity_label'], entity['entity_label'])
                entity_text = entity['entity_text'].lower().strip()
                
                # Build entity type index
                if entity_label not in self.entity_index:
                    self.entity_index[entity_label] = {}
                if entity_text not in self.entity_index[entity_label]:
                    self.entity_index[entity_label][entity_text] = []
                self.entity_index[entity_label][entity_text].append({
                    'file': file_name,
                    'score': entity['score']
                })
                
                # Build document entity index
                if file_name not in self.document_entities:
                    self.document_entities[file_name] = []
                self.document_entities[file_name].append({
                    'label': entity_label,
                    'text': entity_text,
                    'score': entity['score']
                })
    
    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract potential legal entities from query"""
        query_lower = query.lower()
        detected_entities = {}
        
        # Legal section patterns
        section_patterns = [
            r'section\s+(\d+[a-z]*)\s*(ipc|crpc|cpc|nia|poa)',
            r'article\s+(\d+[a-z]*)',
            r'rule\s+(\d+[a-z]*)',
            r'order\s+(\d+[a-z]*)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                if 'STATUTE' not in detected_entities:
                    detected_entities['STATUTE'] = []
                for match in matches:
                    if isinstance(match, tuple):
                        detected_entities['STATUTE'].append(' '.join(match))
                    else:
                        detected_entities['STATUTE'].append(match)
        
        # Court patterns
        court_patterns = [
            r'supreme\s+court', r'high\s+court', r'district\s+court',
            r'sessions?\s+court', r'magistrate', r'tribunal'
        ]
        
        for pattern in court_patterns:
            if re.search(pattern, query_lower):
                if 'COURT' not in detected_entities:
                    detected_entities['COURT'] = []
                matches = re.findall(pattern, query_lower)
                detected_entities['COURT'].extend(matches)
        
        # Case type patterns
        case_patterns = [
            r'criminal\s+appeal', r'civil\s+appeal', r'writ\s+petition',
            r'special\s+leave\s+petition', r'review\s+petition', r'bail\s+application'
        ]
        
        for pattern in case_patterns:
            if re.search(pattern, query_lower):
                if 'CASE_TYPE' not in detected_entities:
                    detected_entities['CASE_TYPE'] = []
                matches = re.findall(pattern, query_lower)
                detected_entities['CASE_TYPE'].extend(matches)
        
        return detected_entities
    
    def _calculate_entity_score(self, doc_name: str, query_entities: Dict[str, List[str]]) -> float:
        """Calculate entity matching score for a document"""
        if not query_entities or doc_name not in self.document_entities:
            return 0.0
        
        doc_entities = self.document_entities[doc_name]
        total_score = 0.0
        total_weight = 0.0
        
        # Entity type weights from config
        entity_weights = getattr(globals(), 'ENTITY_TYPE_WEIGHTS', {
            'STATUTE': 3.0, 'PROVISION': 2.5, 'COURT': 2.0, 'CASE_TYPE': 2.0,
            'JUDGE': 1.5, 'DATE': 1.0, 'OTHER': 0.5
        })
        
        for query_entity_type, query_entity_texts in query_entities.items():
            weight = entity_weights.get(query_entity_type, entity_weights.get('OTHER', 0.5))
            
            for query_text in query_entity_texts:
                query_text_lower = query_text.lower().strip()
                
                # Find matching entities in document
                for doc_entity in doc_entities:
                    if doc_entity['label'] == query_entity_type:
                        doc_text_lower = doc_entity['text'].lower().strip()
                        
                        # Exact match
                        if query_text_lower == doc_text_lower:
                            total_score += weight * doc_entity['score']
                            total_weight += weight
                        # Partial match
                        elif query_text_lower in doc_text_lower or doc_text_lower in query_text_lower:
                            total_score += (weight * 0.7) * doc_entity['score']
                            total_weight += weight * 0.7
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _expand_query_with_entities(self, query: str, query_entities: Dict[str, List[str]]) -> str:
        """Expand query with related legal terms and entities"""
        expanded_terms = [query]
        
        # Add legal synonyms and related terms
        legal_expansions = {
            'murder': ['homicide', 'culpable homicide', 'killing', 'death'],
            'theft': ['stealing', 'larceny', 'misappropriation'],
            'fraud': ['cheating', 'deception', 'misrepresentation'],
            'contract': ['agreement', 'covenant', 'undertaking'],
            'property': ['immovable property', 'real estate', 'land'],
            'appeal': ['appellate', 'revision', 'review'],
            'bail': ['interim bail', 'anticipatory bail', 'regular bail'],
            'divorce': ['dissolution of marriage', 'matrimonial dispute'],
            'custody': ['guardianship', 'ward', 'minor']
        }
        
        query_lower = query.lower()
        for term, expansions in legal_expansions.items():
            if term in query_lower:
                expanded_terms.extend(expansions[:2])  # Add top 2 related terms
        
        # Add entity-specific terms
        for entity_type, entity_texts in query_entities.items():
            if entity_type == 'STATUTE':
                expanded_terms.extend(['indian penal code', 'criminal procedure code', 'constitution'])
            elif entity_type == 'COURT':
                expanded_terms.extend(['judgment', 'order', 'decision'])
        
        return ' '.join(expanded_terms)

    def search_documents(self, query: str, top_k: int = TOP_K, use_hybrid_scoring: bool = True) -> List[Dict]:
        """Search for similar documents using hybrid BGE + NER approach"""
        if not self.index or not self.document_map or not self.embedder:
            st.error("Models not loaded properly")
            return []
            
        try:
            # Step 1: Extract entities from query
            query_entities = self._extract_query_entities(query)
            
            # Step 2: Expand query with related legal terms
            if use_hybrid_scoring and query_entities:
                expanded_query = self._expand_query_with_entities(query, query_entities)
                st.info(f"🔍 Enhanced query with legal entities: {len(query_entities)} entity types detected")
            else:
                expanded_query = query
            
            # Step 3: Get more candidates for re-ranking
            expansion_factor = getattr(globals(), 'SEARCH_EXPANSION_FACTOR', 3)
            search_k = min(top_k * expansion_factor, 50)  # Get more candidates for re-ranking
            
            # Step 4: BGE semantic search
            query_embedding = self.embedder.encode([expanded_query])
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Step 5: Calculate hybrid scores
            candidates = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.document_map):
                    doc_name = self.document_map[idx]
                    
                    # BGE semantic similarity
                    semantic_score = SIM_CONVERT(distance)
                    
                    # Entity matching score
                    entity_score = 0.0
                    if use_hybrid_scoring and query_entities:
                        entity_score = self._calculate_entity_score(doc_name, query_entities)
                    
                    # Hybrid score combination
                    if use_hybrid_scoring and entity_score > 0:
                        # Weighted combination using config weights
                        semantic_weight = getattr(globals(), 'SEMANTIC_WEIGHT', 0.7)
                        entity_weight = getattr(globals(), 'ENTITY_WEIGHT', 0.3)
                        hybrid_score = (semantic_weight * semantic_score) + (entity_weight * entity_score)
                        score_type = "Hybrid (BGE + NER)"
                    else:
                        hybrid_score = semantic_score
                        score_type = "Semantic (BGE)"
                    
                    candidates.append({
                        'doc_name': doc_name,
                        'semantic_score': semantic_score,
                        'entity_score': entity_score,
                        'hybrid_score': hybrid_score,
                        'score_type': score_type,
                        'original_rank': i + 1
                    })
            
            # Step 6: Re-rank by hybrid score
            candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Step 7: Build final results
            results = []
            for i, candidate in enumerate(candidates[:top_k]):
                doc_name = candidate['doc_name']
                
                # Load document content
                doc_path = os.path.join(CLEANED_FOLDER, doc_name)
                content = self._load_document_content(doc_path)
                
                # Get entities for this document
                entities = self._get_document_entities(doc_name)
                
                # Find corresponding PDF
                pdf_path = self._find_pdf_path(doc_name)
                
                results.append({
                    'rank': i + 1,
                    'filename': doc_name,
                    'similarity_score': candidate['hybrid_score'],
                    'semantic_score': candidate['semantic_score'],
                    'entity_score': candidate['entity_score'],
                    'score_type': candidate['score_type'],
                    'original_rank': candidate['original_rank'],
                    'content': content,
                    'entities': entities,
                    'pdf_path': pdf_path,
                    'query_entities': query_entities
                })
                    
            return results
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []
    
    def _load_document_content(self, doc_path: str) -> str:
        """Load document content from cleaned texts"""
        try:
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                max_length = getattr(globals(), 'MAX_CONTENT_LENGTH', 500)
                return content[:max_length] + "..." if len(content) > max_length else content
            else:
                return "Content not available"
        except Exception as e:
            return f"Error loading content: {str(e)}"
    
    def _get_document_entities(self, doc_name: str) -> List[Dict]:
        """Get top entities for a document"""
        if not self.ner_data:
            return []
            
        doc_entities = [entity for entity in self.ner_data if entity['file'] == doc_name]
        
        # Group entities by type and get top ones
        entity_groups = {}
        confidence_threshold = getattr(globals(), 'ENTITY_CONFIDENCE_THRESHOLD', 0.8)
        
        for entity in doc_entities:
            if entity['score'] > confidence_threshold:  # Filter by confidence
                label = LABEL_MAPPING.get(entity['entity_label'], entity['entity_label'])
                if label not in entity_groups:
                    entity_groups[label] = []
                entity_groups[label].append({
                    'text': entity['entity_text'],
                    'score': entity['score']
                })
        
        # Get top entities per type
        max_per_type = getattr(globals(), 'MAX_ENTITIES_PER_TYPE', 3)
        max_total = getattr(globals(), 'MAX_ENTITIES_PER_DOC', 10)
        
        top_entities = []
        for label, entities in entity_groups.items():
            sorted_entities = sorted(entities, key=lambda x: x['score'], reverse=True)[:max_per_type]
            for entity in sorted_entities:
                top_entities.append({
                    'label': label,
                    'text': entity['text'],
                    'score': entity['score']
                })
        
        return sorted(top_entities, key=lambda x: x['score'], reverse=True)[:max_total]
    
    def _find_pdf_path(self, doc_name: str) -> Optional[str]:
        """Find corresponding PDF file"""
        # Extract base name without extension
        base_name = doc_name.replace('.txt', '')
        
        # Search in PDF folders
        pdf_folders = [
            os.path.join(PDF_FOLDER, "manupatra"),
            os.path.join(PDF_FOLDER, "sci gov")
        ]
        
        for folder in pdf_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith('.pdf'):
                        # Simple matching - you might need to adjust this based on your naming convention
                        if base_name.split('_')[0] in file or file.replace('.pdf', '') in base_name:
                            return os.path.join(folder, file)
        
        return None

def highlight_entities(text: str, entities: List[Dict]) -> str:
    """Highlight entities in text with colors"""
    highlighted_text = text
    
    # Sort entities by position to avoid overlap issues
    sorted_entities = sorted(entities, key=lambda x: len(x['text']), reverse=True)
    
    for entity in sorted_entities:
        entity_text = entity['text']
        label = entity['label']
        color = ENTITY_COLORS.get(label, "#CCCCCC")
        
        # Create highlighted span
        highlighted_span = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;" title="{label} (Score: {entity["score"]:.2f})">{entity_text}</span>'
        
        # Replace in text (case insensitive)
        pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
        highlighted_text = pattern.sub(highlighted_span, highlighted_text, count=1)
    
    return highlighted_text

def create_pdf_download_link(pdf_path: str, filename: str) -> str:
    """Create a download link for PDF"""
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" target="_blank">📄 Download PDF</a>'
        return href
    return "📄 PDF not available"

def main():
    st.set_page_config(
        page_title="Legal Document Retrieval System",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .entity-tag {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .similarity-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Retrieval System</h1>', unsafe_allow_html=True)
    
    # Simple status indicator
    if st.session_state.get('system_loaded', False):
        st.success("🟢 System Ready - Search through legal documents using AI-powered hybrid search")
    else:
        st.error("🔴 System Loading Failed - Please refresh the page")
    
    # Initialize retriever and auto-load system
    retriever = LegalDocumentRetriever()
    
    # Auto-load system on startup (only once)
    if 'system_loaded' not in st.session_state:
        with st.spinner("🔄 Loading legal document system..."):
            success = retriever.load_models_and_data()
            st.session_state.system_loaded = success
            if success:
                st.session_state.retriever = retriever
            else:
                st.error("❌ Failed to load system. Please refresh the page.")
                st.stop()
    else:
        retriever = st.session_state.retriever
    
    # Main content - single column layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search interface
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.subheader("🔍 Search Legal Documents")
        
        query = st.text_input(
            "Enter your legal query:",
            placeholder="e.g., Section 302 IPC murder appeal, contract breach damages, property dispute...",
            help="Enter keywords, legal sections, case types, or any legal query"
        )
        
        # Enable hybrid scoring by default (hidden from user)
        use_hybrid = True
        show_debug = False
        
        col_search, col_clear = st.columns([1, 1])
        with col_search:
            search_button = st.button("🔍 Search Documents", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Search results
        if search_button and query:
                with st.spinner("Searching legal documents..."):
                    results = retriever.search_documents(query, TOP_K, use_hybrid_scoring=use_hybrid)
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant legal documents")
                    
                    for result in results:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        # Header with rank and filename
                        col_header, col_score = st.columns([3, 1])
                        with col_header:
                            # Clean filename display
                            clean_filename = result['filename'].replace('.txt', '').replace('_', ' ')
                            st.markdown(f"### 📄 {result['rank']}. {clean_filename}")
                        with col_score:
                            # Simple relevance indicator
                            score = result["similarity_score"]
                            if score > 0.8:
                                st.markdown('🟢 **Highly Relevant**')
                            elif score > 0.6:
                                st.markdown('🟡 **Very Relevant**')
                            elif score > 0.4:
                                st.markdown('🟠 **Relevant**')
                            else:
                                st.markdown('🔵 **Somewhat Relevant**')
                        
                        # Content preview
                        st.markdown("**📖 Content Preview:**")
                        if result['entities']:
                            highlighted_content = highlight_entities(result['content'], result['entities'])
                            st.markdown(highlighted_content, unsafe_allow_html=True)
                        else:
                            st.text(result['content'])
                        
                        # Entities - simplified display
                        if result['entities']:
                            st.markdown("**🏷️ Key Legal Terms:**")
                            entity_html = ""
                            # Show only the most important entity types
                            important_entities = [e for e in result['entities'][:6] 
                                                if e['label'] in ['STATUTE', 'COURT', 'PROVISION', 'JUDGE', 'CASE_NUMBER']]
                            
                            for entity in important_entities:
                                color = ENTITY_COLORS.get(entity['label'], "#CCCCCC")
                                entity_html += f'<span class="entity-tag" style="background-color: {color}; padding: 3px 8px; margin: 2px; border-radius: 12px; font-size: 0.85rem;">{entity["text"]}</span> '
                            
                            if entity_html:
                                st.markdown(entity_html, unsafe_allow_html=True)
                        
                        # PDF link
                        if result['pdf_path']:
                            pdf_link = create_pdf_download_link(result['pdf_path'], result['filename'].replace('.txt', '.pdf'))
                            st.markdown(pdf_link, unsafe_allow_html=True)
                        else:
                            st.markdown("📄 PDF not available")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")
                        
                else:
                    st.warning("⚠️ No relevant documents found. Try different keywords.")
    
    with col2:
        # Query suggestions
        st.subheader("💡 Quick Search")
        
        # Show only top 6 most common legal queries
        top_suggestions = [
            "Section 302 IPC murder conviction",
            "Contract breach damages", 
            "Property dispute rights",
            "Criminal appeal bail",
            "Constitutional Article 14",
            "Sexual Assault"
            "Service tax penalty"
        ]
        
        for suggestion in top_suggestions:
            if st.button(f"🔍 {suggestion}", key=suggestion, use_container_width=True):
                st.session_state.query = suggestion
                st.rerun()

if __name__ == "__main__":
    main()