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

# Import configuration
try:
    from config import *
except ImportError:
    # Fallback configuration
    INDEX_PATH = "train_faiss_index_legalbert.idx"
    DOC_MAP_PATH = "train_document_map_legalbert.npy"
    CLEANED_FOLDER = "cleaned_texts"
    PDF_FOLDER = "nlp ds"
    NER_OUTPUT_PATH = "legal_ner_output.json"
    EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
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

class SimpleLegalRetriever:
    def __init__(self):
        self.index = None
        self.document_map = None
        self.embedder = None
        self.ner_data = None
        self.entity_index = {}
        self.document_entities = {}
        
    def load_system(self):
        """Load all system components silently"""
        try:
            # Load FAISS index
            if os.path.exists(INDEX_PATH):
                self.index = faiss.read_index(INDEX_PATH)
            else:
                return False
                
            # Load document map
            if os.path.exists(DOC_MAP_PATH):
                self.document_map = np.load(DOC_MAP_PATH, allow_pickle=True)
            else:
                return False
                
            # Load sentence transformer
            self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
                
            # Load NER data
            if os.path.exists(NER_OUTPUT_PATH):
                with open(NER_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    self.ner_data = json.load(f)
                self._build_entity_indexes()
            else:
                self.ner_data = []
                
            return True
            
        except Exception as e:
            return False
    
    def _build_entity_indexes(self):
        """Build entity indexes for hybrid search"""
        self.entity_index = {}
        self.document_entities = {}
        
        for entity in self.ner_data:
            if entity['score'] > 0.8:
                file_name = entity['file']
                entity_label = LABEL_MAPPING.get(entity['entity_label'], entity['entity_label'])
                entity_text = entity['entity_text'].lower().strip()
                
                if file_name not in self.document_entities:
                    self.document_entities[file_name] = []
                self.document_entities[file_name].append({
                    'label': entity_label,
                    'text': entity_text,
                    'score': entity['score']
                })
    
    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal entities from query"""
        query_lower = query.lower()
        detected_entities = {}
        
        # Legal section patterns
        section_patterns = [
            r'section\s+(\d+[a-z]*)\s*(ipc|crpc|cpc|nia|poa)',
            r'article\s+(\d+[a-z]*)',
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
        court_patterns = [r'supreme\s+court', r'high\s+court', r'district\s+court']
        for pattern in court_patterns:
            if re.search(pattern, query_lower):
                if 'COURT' not in detected_entities:
                    detected_entities['COURT'] = []
                matches = re.findall(pattern, query_lower)
                detected_entities['COURT'].extend(matches)
        
        return detected_entities
    
    def _calculate_entity_score(self, doc_name: str, query_entities: Dict[str, List[str]]) -> float:
        """Calculate entity matching score"""
        if not query_entities or doc_name not in self.document_entities:
            return 0.0
        
        doc_entities = self.document_entities[doc_name]
        total_score = 0.0
        total_weight = 0.0
        
        entity_weights = {'STATUTE': 3.0, 'COURT': 2.0, 'JUDGE': 1.5, 'OTHER': 0.5}
        
        for query_entity_type, query_entity_texts in query_entities.items():
            weight = entity_weights.get(query_entity_type, 0.5)
            
            for query_text in query_entity_texts:
                query_text_lower = query_text.lower().strip()
                
                for doc_entity in doc_entities:
                    if doc_entity['label'] == query_entity_type:
                        doc_text_lower = doc_entity['text'].lower().strip()
                        
                        if query_text_lower == doc_text_lower:
                            total_score += weight * doc_entity['score']
                            total_weight += weight
                        elif query_text_lower in doc_text_lower or doc_text_lower in query_text_lower:
                            total_score += (weight * 0.7) * doc_entity['score']
                            total_weight += weight * 0.7
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def search_documents(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Search documents with hybrid BGE + NER"""
        if self.index is None or self.document_map is None or self.embedder is None:
            return []
            
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query)
            
            # BGE semantic search
            query_embedding = self.embedder.encode([query])
            search_k = min(top_k * 2, 20)
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Calculate hybrid scores
            candidates = []
            
            # Ensure we have valid results
            if len(distances) > 0 and len(indices) > 0:
                dist_array = distances[0] if len(distances.shape) > 1 else distances
                idx_array = indices[0] if len(indices.shape) > 1 else indices
                
                for i, (distance, idx) in enumerate(zip(dist_array, idx_array)):
                    if idx >= 0 and idx < len(self.document_map):
                        doc_name = self.document_map[idx]
                        
                        semantic_score = SIM_CONVERT(float(distance))
                        entity_score = self._calculate_entity_score(doc_name, query_entities)
                        
                        # Hybrid scoring: 70% semantic + 30% entity
                        if entity_score > 0:
                            hybrid_score = (0.7 * semantic_score) + (0.3 * entity_score)
                        else:
                            hybrid_score = semantic_score
                        
                        candidates.append({
                            'doc_name': doc_name,
                            'hybrid_score': hybrid_score,
                            'entity_boost': entity_score > 0
                        })
            
            # Re-rank by hybrid score
            candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # Build final results
            results = []
            for i, candidate in enumerate(candidates[:top_k]):
                doc_name = candidate['doc_name']
                
                # Load document content
                doc_path = os.path.join(CLEANED_FOLDER, doc_name)
                content = self._load_document_content(doc_path)
                
                # Get entities
                entities = self._get_document_entities(doc_name)
                
                # Find PDF
                pdf_path = self._find_pdf_path(doc_name)
                
                results.append({
                    'rank': i + 1,
                    'filename': doc_name,
                    'similarity_score': candidate['hybrid_score'],
                    'entity_boost': candidate['entity_boost'],
                    'content': content,
                    'entities': entities,
                    'pdf_path': pdf_path
                })
                    
            return results
            
        except Exception as e:
            return []
    
    def _load_document_content(self, doc_path: str) -> str:
        """Load document content"""
        try:
            if os.path.exists(doc_path):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content[:400] + "..." if len(content) > 400 else content
            else:
                return "Content not available"
        except Exception as e:
            return f"Error loading content"
    
    def _get_document_entities(self, doc_name: str) -> List[Dict]:
        """Get top entities for document"""
        if not self.ner_data:
            return []
            
        doc_entities = [entity for entity in self.ner_data if entity['file'] == doc_name]
        
        # Get top entities
        entity_groups = {}
        for entity in doc_entities:
            if entity['score'] > 0.8:
                label = LABEL_MAPPING.get(entity['entity_label'], entity['entity_label'])
                if label not in entity_groups:
                    entity_groups[label] = []
                entity_groups[label].append({
                    'text': entity['entity_text'],
                    'score': entity['score']
                })
        
        top_entities = []
        for label, entities in entity_groups.items():
            sorted_entities = sorted(entities, key=lambda x: x['score'], reverse=True)[:2]
            for entity in sorted_entities:
                top_entities.append({
                    'label': label,
                    'text': entity['text'],
                    'score': entity['score']
                })
        
        return sorted(top_entities, key=lambda x: x['score'], reverse=True)[:6]
    
    def _find_pdf_path(self, doc_name: str) -> Optional[str]:
        """Find corresponding PDF"""
        base_name = doc_name.replace('.txt', '')
        pdf_folders = [
            os.path.join(PDF_FOLDER, "manupatra"),
            os.path.join(PDF_FOLDER, "sci gov")
        ]
        
        for folder in pdf_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith('.pdf'):
                        if base_name.split('_')[0] in file or file.replace('.pdf', '') in base_name:
                            return os.path.join(folder, file)
        return None

def create_pdf_link(pdf_path: str, filename: str) -> str:
    """Create PDF download link"""
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        b64_pdf = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" target="_blank" style="text-decoration: none; background-color: #0066cc; color: white; padding: 8px 16px; border-radius: 4px; font-size: 14px;">📄 Download PDF</a>'
        return href
    return '<span style="color: #666;">📄 PDF not available</span>'

def main():
    st.set_page_config(
        page_title="Legal Document Search",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for clean design
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    .search-box {
        font-size: 1.1rem;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        width: 100%;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f4e79;
    }
    .entity-tag {
        display: inline-block;
        padding: 4px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Search</h1>', unsafe_allow_html=True)
    
    # Initialize and load system silently
    if 'retriever' not in st.session_state:
        retriever = SimpleLegalRetriever()
        success = retriever.load_system()
        if success:
            st.session_state.retriever = retriever
            st.session_state.system_ready = True
        else:
            st.error("❌ System unavailable. Please refresh the page.")
            st.stop()
    
    retriever = st.session_state.retriever
    
    # No status message - keep it clean
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🔍 Search Legal Documents")
        
        # Search input
        query = st.text_input(
            "",
            placeholder="Enter your legal query (e.g., landlord crime hearing, Section 302 IPC, contract breach...)",
            key="search_query"
        )
        
        # Search button
        search_clicked = st.button("🔍 Search Documents", type="primary", use_container_width=True)
        
        # Search results
        if search_clicked and query:
            with st.spinner("Searching legal documents..."):
                results = retriever.search_documents(query, TOP_K)
            
            if results:
                st.success(f"✅ Found {len(results)} relevant documents")
                
                for result in results:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # Header
                    col_title, col_relevance = st.columns([3, 1])
                    with col_title:
                        clean_name = result['filename'].replace('.txt', '').replace('_', ' ')
                        st.markdown(f"**📄 {result['rank']}. {clean_name}**")
                    
                    with col_relevance:
                        score = result['similarity_score']
                        if result['entity_boost']:
                            st.markdown("🚀 **Enhanced Match**")
                        elif score > 0.7:
                            st.markdown("🟢 **High Relevance**")
                        elif score > 0.5:
                            st.markdown("🟡 **Good Match**")
                        else:
                            st.markdown("🔵 **Related**")
                    
                    # Content
                    st.markdown("**📖 Content Preview:**")
                    st.text(result['content'])
                    
                    # Entities
                    if result['entities']:
                        st.markdown("**🏷️ Legal Terms:**")
                        entity_html = ""
                        for entity in result['entities']:
                            color = ENTITY_COLORS.get(entity['label'], "#666666")
                            entity_html += f'<span class="entity-tag" style="background-color: {color};">{entity["text"]}</span> '
                        st.markdown(entity_html, unsafe_allow_html=True)
                    
                    # PDF link
                    if result['pdf_path']:
                        pdf_link = create_pdf_link(result['pdf_path'], result['filename'].replace('.txt', '.pdf'))
                        st.markdown(pdf_link, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.warning("⚠️ No relevant documents found. Try different keywords or check spelling.")
    
    with col2:
        st.markdown("### 💡 Quick Search")
        
        suggestions = [
            "landlord crime hearing",
            "Section 302 IPC murder",
            "contract breach damages",
            "property dispute rights",
            "criminal appeal bail",
            "divorce custody child"
        ]
        
        for suggestion in suggestions:
            if st.button(f"🔍 {suggestion}", key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.search_query = suggestion
                st.rerun()

if __name__ == "__main__":
    main()