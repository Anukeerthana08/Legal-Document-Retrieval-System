# 📖 Legal Document Retrieval System - Usage Guide

This comprehensive guide will help you get the most out of the Legal Document Retrieval System.

## 🚀 Getting Started

### Step 1: Installation
```bash
# Option 1: Automated installation
python install.py

# Option 2: Manual installation
pip install -r requirements.txt
```

### Step 2: System Check
```bash
# Verify all components are working
python test_system.py
```

### Step 3: Launch Application
```bash
# Option 1: Using the launcher
python run_app.py

# Option 2: Direct launch
streamlit run legal_retrieval_app.py

# Option 3: Using startup script (after installation)
./start_app.sh  # Linux/Mac
start_app.bat   # Windows
```

## 🔍 Search Strategies

### Basic Search
- **Simple Keywords**: `murder`, `contract`, `property`
- **Legal Sections**: `Section 302 IPC`, `Article 14`, `Section 138 NI Act`
- **Case Types**: `criminal appeal`, `civil suit`, `writ petition`

### Advanced Search Techniques

#### 1. Combining Legal Terms
```
Section 302 IPC murder conviction appeal
Contract breach specific performance damages
Property dispute adverse possession title
```

#### 2. Using Legal Concepts
```
Constitutional fundamental rights violation
Criminal conspiracy common intention
Tort negligence duty of care
```

#### 3. Procedural Queries
```
Bail application grounds criminal appeal
Summary judgment application civil procedure
Interim injunction balance of convenience
```

#### 4. Subject-Specific Searches
```
# Corporate Law
Company law director liability breach
Merger acquisition regulatory approval
Insider trading securities violation

# Family Law
Divorce mutual consent maintenance
Child custody best interest welfare
Domestic violence protection order

# Tax Law
Income tax assessment penalty
GST input credit reversal
Service tax exemption notification
```

## 🏷️ Understanding Entity Types

### Court Entities
- **COURT**: Supreme Court, High Court, District Court
- **JUDGE**: Justice names, bench compositions
- **CASE_NUMBER**: Case registration numbers, appeal numbers

### Party Entities
- **PETITIONER**: Plaintiffs, appellants, applicants
- **RESPONDENT**: Defendants, respondents, opposite parties
- **LAWYER**: Counsel names, senior advocates

### Legal Framework
- **STATUTE**: Acts, codes, regulations (IPC, CPC, Constitution)
- **PROVISION**: Specific sections, articles, rules
- **PRECEDENT**: Cited case laws, landmark judgments

### Contextual Information
- **DATE**: Judgment dates, incident dates, filing dates
- **ORG**: Government bodies, corporations, institutions
- **GPE**: States, cities, jurisdictions
- **OTHER_PERSON**: Witnesses, officials, third parties

## 📊 Interpreting Results

### Similarity Scores
- **0.8-1.0**: Highly relevant, exact matches
- **0.6-0.8**: Very relevant, strong semantic similarity
- **0.4-0.6**: Moderately relevant, related concepts
- **0.2-0.4**: Somewhat relevant, tangential connection
- **0.0-0.2**: Low relevance, weak connection

### Content Preview
- First 500 characters of the document
- Key entities highlighted in color
- Context around search terms

### Entity Highlighting
- Hover over highlighted text to see:
  - Entity type
  - Confidence score
  - Position in document

## 🎯 Search Examples by Legal Domain

### Criminal Law
```
# Murder Cases
Section 302 IPC murder conviction life imprisonment
Culpable homicide murder distinction intention knowledge
Self defense private defense right protection

# Theft and Property Crimes
Section 379 IPC theft dishonest intention
Criminal breach of trust Section 406 IPC
Cheating Section 420 IPC fraud deception

# Procedural Criminal Law
Criminal appeal conviction sentence reduction
Bail application Section 437 CrPC grounds
Anticipatory bail Section 438 CrPC conditions
```

### Civil Law
```
# Contract Law
Breach of contract damages specific performance
Frustration of contract impossibility performance
Consideration contract validity enforceability

# Property Law
Adverse possession title limitation period
Easement right of way dominant servient
Partition joint property co-ownership rights

# Tort Law
Negligence duty of care breach damages
Defamation libel slander reputation harm
Nuisance private public interference enjoyment
```

### Constitutional Law
```
# Fundamental Rights
Article 14 equality before law discrimination
Article 19 freedom speech expression reasonable restrictions
Article 21 life personal liberty due process

# Writ Jurisdiction
Mandamus public duty enforcement
Certiorari judicial review administrative action
Habeas corpus illegal detention custody
```

### Commercial Law
```
# Company Law
Director liability breach fiduciary duty
Oppression mismanagement minority shareholders
Winding up company insolvency proceedings

# Banking and Finance
Negotiable Instruments Act dishonor cheque
Recovery of debts banks financial institutions
Insolvency bankruptcy resolution process
```

## 🔧 Customization Options

### Modifying Search Parameters
Edit `config.py` to customize:
```python
TOP_K = 10  # Show more results
ENTITY_CONFIDENCE_THRESHOLD = 0.7  # Lower threshold for more entities
MAX_CONTENT_LENGTH = 800  # Longer content previews
```

### Adding Custom Entity Types
```python
LABEL_MAPPING["LABEL_13"] = "CUSTOM_ENTITY"
ENTITY_COLORS["CUSTOM_ENTITY"] = "#FF5733"
```

### Custom Query Suggestions
```python
QUERY_SUGGESTIONS.extend([
    "Your custom legal query 1",
    "Your custom legal query 2"
])
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No relevant documents found"
**Solutions:**
- Try broader keywords
- Use synonyms or alternative terms
- Check spelling and legal terminology
- Reduce specificity of query

#### 2. Low similarity scores
**Causes:**
- Query too specific or narrow
- Mismatch between query and document language
- Technical legal terms not in corpus

**Solutions:**
- Use more general legal concepts
- Try different phrasings
- Include context words

#### 3. Missing entities or poor highlighting
**Causes:**
- Low confidence scores
- Entity recognition limitations
- Text preprocessing issues

**Solutions:**
- Lower confidence threshold in config
- Check original document quality
- Report systematic issues

#### 4. Slow performance
**Causes:**
- Large document corpus
- Insufficient memory
- Model loading delays

**Solutions:**
- Ensure 8GB+ RAM available
- Close other applications
- Use smaller TOP_K values

### Performance Optimization

#### Memory Management
```python
# In config.py
TOP_K = 3  # Reduce results
MAX_ENTITIES_PER_DOC = 5  # Fewer entities
ENTITY_CONFIDENCE_THRESHOLD = 0.9  # Higher threshold
```

#### Search Optimization
- Use specific legal terms
- Avoid very long queries
- Combine related concepts
- Use established legal phrases

## 📈 Best Practices

### Query Formulation
1. **Start Specific**: Use exact legal provisions when known
2. **Add Context**: Include case type, jurisdiction, or domain
3. **Use Legal Language**: Employ standard legal terminology
4. **Iterate**: Refine queries based on initial results

### Result Analysis
1. **Check Similarity Scores**: Focus on scores > 0.5
2. **Review Entities**: Verify relevant legal entities are highlighted
3. **Read Previews**: Scan content for relevance indicators
4. **Access Full Documents**: Use PDF links for complete analysis

### System Usage
1. **Regular Updates**: Reload system when data changes
2. **Monitor Performance**: Check system status in sidebar
3. **Save Queries**: Note effective search patterns
4. **Report Issues**: Document systematic problems

## 🔮 Advanced Features

### Batch Processing
For multiple queries, consider scripting:
```python
queries = [
    "Section 302 IPC murder",
    "Contract breach damages",
    "Property dispute title"
]

for query in queries:
    results = retriever.search_documents(query)
    # Process results
```

### Custom Filtering
Filter results by entity types:
```python
# Show only cases with specific judges
judge_cases = [r for r in results if any(e['label'] == 'JUDGE' for e in r['entities'])]

# Filter by date entities
recent_cases = [r for r in results if any('2023' in e['text'] for e in r['entities'])]
```

### Export Functionality
Save search results:
```python
import json
with open('search_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

## 📞 Support and Resources

### Getting Help
1. **Check README.md**: Comprehensive setup guide
2. **Run test_system.py**: Diagnose technical issues
3. **Review config.py**: Understand customization options
4. **Check console output**: Look for error messages

### Legal Research Tips
1. **Know Your Jurisdiction**: Focus on relevant court levels
2. **Understand Legal Hierarchy**: Supreme Court > High Court > Lower Courts
3. **Check Citation Formats**: Use standard legal citations
4. **Verify Currency**: Ensure legal provisions are current

### System Maintenance
1. **Regular Backups**: Save important configurations
2. **Update Dependencies**: Keep packages current
3. **Monitor Performance**: Track search effectiveness
4. **Document Changes**: Record customizations

---

**⚖️ Legal Document Retrieval System** - Empowering legal research through intelligent document discovery.