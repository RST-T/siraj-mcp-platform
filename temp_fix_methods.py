def _generate_hadith_methodology_section(self, root: str, hadith_data: dict) -> str:
    """Generate Hadith analysis methodology based on database findings"""
    references = hadith_data.get('references', 0)
    
    if references > 0:
        return f"""2. **Hadith Analysis (DATABASE-INFORMED)**
   - **Database Status**: Found {references} references of root '{root}' in Hadith corpus
   - **Methodology**: Analyze narrator chains (isnad) and authenticity grades from database
   - **Subject Mapping**: Use database subject tags for thematic analysis
   - **Cross-Validation**: Compare database authenticity grades with traditional classifications"""
    else:
        return f"""2. **Hadith Analysis (DATABASE-GUIDED)**
   - **Database Status**: No references of root '{root}' found in Hadith corpus
   - **Methodology**: Search for morphologically related forms using database patterns
   - **Validation**: Cross-reference with traditional Hadith scholarship"""

def _generate_classical_methodology_section(self, root: str, classical_data: dict) -> str:
    """Generate classical Arabic analysis methodology based on database findings"""
    examples = classical_data.get('usage_examples', 0)
    
    if examples > 0:
        return f"""3. **Classical Arabic Analysis (DATABASE-INFORMED)**
   - **Database Status**: Found {examples} usage examples of root '{root}' in classical texts
   - **Methodology**: Analyze literary contexts and cultural markers from database
   - **Diachronic Analysis**: Trace meaning evolution using database temporal data
   - **Genre Analysis**: Apply database genre classifications for usage pattern mapping"""
    else:
        return f"""3. **Classical Arabic Analysis (DATABASE-GUIDED)**
   - **Database Status**: No usage examples of root '{root}' found in classical texts
   - **Methodology**: Use database morphological patterns to guide external research
   - **Pattern Matching**: Apply successful analysis patterns from database roots"""