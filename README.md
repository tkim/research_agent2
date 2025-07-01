# Research Agent 2

An intelligent assistant designed to conduct thorough research and analysis using web search capabilities and various APIs. The Research Agent operates locally on your system and is designed to be helpful, accurate, and efficient while respecting computational limitations.

## Features

### Core Capabilities

1. **Web Search and Information Gathering**
   - Multi-engine web search (Google, Bing, DuckDuckGo)
   - Content scraping and analysis
   - Source verification and credibility assessment
   - Automatic citation generation

2. **Research Methodology**
   - Query analysis and decomposition
   - Research strategy development
   - Information synthesis from multiple sources
   - Gap identification and recommendations

3. **API Integration**
   - Flexible API client system
   - Rate limiting and caching
   - Built-in support for News API, Wikipedia, arXiv
   - Easy extension for new APIs

4. **Citation Management**
   - Multiple citation styles (APA, MLA, Chicago, IEEE, etc.)
   - Automatic metadata extraction
   - Bibliography generation
   - Export to JSON, BibTeX formats

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/tkim/research_agent2.git
cd research_agent2

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install in development mode with extra dependencies
pip install -e ".[dev]"
```

## Configuration

Copy the example configuration and customize it:

```bash
cp config.json config_local.json
```

Edit `config_local.json` to add your API keys:

```json
{
  "apis": {
    "newsapi": {
      "auth_config": {
        "api_key": "your_news_api_key_here"
      }
    }
  }
}
```

## Usage

### Command Line Interface

#### Interactive Mode

```bash
python main.py --interactive
```

This launches an interactive session where you can:
- Conduct research: `research <your question>`
- Perform web search: `search <search terms>`
- Generate bibliography: `bibliography`
- Export citations: `export`

#### Direct Research

```bash
# Conduct research on a specific topic
python main.py --research "artificial intelligence applications in healthcare"

# Perform a web search
python main.py --search "climate change 2024" --num-results 10

# Save results to file
python main.py --research "quantum computing breakthroughs" --output results.json
```

#### Citation Styles

```bash
# Generate bibliography in different styles
python main.py --research "machine learning ethics" --citation-style apa
python main.py --research "renewable energy trends" --citation-style ieee
```

### Python API

```python
import asyncio
from research_agent import ResearchAgent
from citation_manager import CitationManager, CitationStyle

async def main():
    # Initialize the research agent
    agent = ResearchAgent()
    
    # Conduct research
    result = await agent.conduct_research(
        "What are the latest developments in renewable energy?"
    )
    
    # Print results
    print(f"Confidence: {result.confidence_level:.2f}")
    print(f"Sources found: {len(result.sources)}")
    print(f"Synthesis: {result.synthesis}")
    
    # Generate citations
    citation_manager = CitationManager()
    for source in result.sources:
        citation_manager.add_source(
            url=source.url,
            title=source.title,
            content=source.content
        )
    
    # Generate bibliography
    bibliography = citation_manager.generate_bibliography(CitationStyle.APA)
    print(bibliography)

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

The Research Agent consists of several modular components:

- **`research_agent.py`** - Core research orchestration and methodology
- **`web_search.py`** - Multi-engine web search and content scraping
- **`research_methodology.py`** - Query analysis and research strategies  
- **`api_integrations.py`** - Flexible API client framework
- **`citation_manager.py`** - Citation generation and bibliography management
- **`main.py`** - Command-line interface and application entry point

## API Keys and Configuration

### Supported APIs

1. **Google Custom Search** (optional)
   - Requires: API key and Search Engine ID
   - Used for: High-quality web search results

2. **Bing Search** (optional)  
   - Requires: API key
   - Used for: Alternative web search results

3. **News API** (optional)
   - Requires: API key
   - Used for: Current news and events

4. **Wikipedia API** (no key required)
   - Used for: Encyclopedia content and summaries

5. **arXiv API** (no key required)
   - Used for: Academic papers and preprints

### Getting API Keys

- **Google Custom Search**: [Google Developers Console](https://developers.google.com/custom-search/v1/introduction)
- **Bing Search**: [Microsoft Azure Portal](https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/)
- **News API**: [NewsAPI.org](https://newsapi.org/)

## Examples

### Research Academic Topics

```bash
python main.py --research "machine learning interpretability methods 2024"
```

### Current Events Research

```bash
python main.py --research "latest developments in climate change policy"
```

### Technical Documentation Search

```bash
python main.py --search "docker container security best practices" --num-results 15
```

### Comparative Analysis

```bash
python main.py --research "compare renewable energy sources efficiency 2024"
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 research_agent2/
black research_agent2/

# Type checking
mypy research_agent2/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with respect for web scraping ethics and robots.txt
- Implements academic citation standards
- Designed for educational and research purposes

## Roadmap

- [ ] Enhanced NLP for better query understanding
- [ ] Support for more citation styles
- [ ] Integration with academic databases
- [ ] Web interface
- [ ] Collaborative research features
- [ ] Export to research management tools

## Support

For support, please:
1. Check the [documentation](https://github.com/tkim/research_agent2/wiki)
2. Search [existing issues](https://github.com/tkim/research_agent2/issues)
3. Create a [new issue](https://github.com/tkim/research_agent2/issues/new) if needed

---

**Note**: This tool is designed for educational and research purposes. Please respect website terms of service, rate limits, and copyright when using the web search capabilities.