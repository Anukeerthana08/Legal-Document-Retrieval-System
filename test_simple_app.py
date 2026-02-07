#!/usr/bin/env python3
"""
Quick test for the simple legal app to verify it works without errors
"""

def test_simple_search():
    """Test the simple app search functionality"""
    print("🧪 Testing Simple Legal App")
    print("=" * 40)
    
    try:
        from simple_legal_app import SimpleLegalRetriever
        
        # Initialize retriever
        retriever = SimpleLegalRetriever()
        
        # Load system
        print("📚 Loading system...")
        success = retriever.load_system()
        
        if not success:
            print("❌ Failed to load system")
            return False
        
        print("✅ System loaded successfully!")
        
        # Test search
        query = "landlord crime hearing"
        print(f"🔍 Testing query: '{query}'")
        
        results = retriever.search_documents(query, 3)
        
        if results:
            print(f"✅ Search successful! Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['filename'][:50]}... (Score: {result['similarity_score']:.3f})")
                if result['entity_boost']:
                    print(f"      🚀 Enhanced with entity matching")
        else:
            print("⚠️ No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_search()
    
    if success:
        print("\n🎉 Simple app test passed!")
        print("🌐 Access the app at: http://localhost:8501")
        print("💡 Try searching for: 'landlord crime hearing'")
    else:
        print("\n❌ Test failed - check errors above")