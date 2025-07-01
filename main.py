#!/usr/bin/env python3
"""
Research Agent 2 - Main Entry Point

This is the main script for running the Research Agent 2 system.
It provides a command-line interface for conducting research operations.
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from research_agent import ResearchAgent
from enhanced_research_agent import EnhancedResearchAgent, ResearchRequest, conduct_research
from web_search import EnhancedWebSearch
from api_integrations import create_research_api_manager
from citation_manager import CitationManager, CitationStyle


def setup_logging(config: Dict[str, Any]):
    """Set up logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[]
    )
    
    logger = logging.getLogger()
    
    # Console handler
    if 'console' in log_config.get('handlers', ['console']):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(console_handler)
    
    # File handler
    if 'file' in log_config.get('handlers', []):
        file_path = log_config.get('file_path', 'research_agent.log')
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default settings.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


class ResearchAgentCLI:
    """Command-line interface for the Research Agent."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResearchAgentCLI")
        
        # Initialize components
        self.research_agent = ResearchAgent(config.get('research_agent2', {}))
        self.enhanced_agent = EnhancedResearchAgent(config.get('research_agent2', {}))
        self.web_search = EnhancedWebSearch(config.get('web_search', {}))
        self.api_manager = create_research_api_manager(self._extract_api_keys())
        self.citation_manager = CitationManager()
        
    def _extract_api_keys(self) -> Dict[str, str]:
        """Extract API keys from configuration."""
        api_keys = {}
        apis_config = self.config.get('apis', {})
        
        for api_name, api_config in apis_config.items():
            auth_config = api_config.get('auth_config', {})
            if 'api_key' in auth_config and auth_config['api_key']:
                api_keys[f'{api_name}_api_key'] = auth_config['api_key']
                
        return api_keys
        
    async def conduct_research(self, query: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive research on a query."""
        self.logger.info(f"Starting enhanced research on: {query}")
        
        try:
            # Create research request for enhanced agent
            request = ResearchRequest(
                query=query,
                session_id=options.get('session_id', 'cli_session'),
                research_type=options.get('research_type', 'comprehensive'),
                max_sources=options.get('max_sources', 15),
                preferred_citation_style=options.get('citation_style', 'apa'),
                quality_threshold=options.get('quality_threshold', 0.6)
            )
            
            # Use enhanced research agent
            result = await self.enhanced_agent.conduct_enhanced_research(request)
            
            return {
                'query': query,
                'response_type': result.get('response_type'),
                'result': result,
                'enhanced': True,
                'timestamp': result.get('timestamp')
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced research failed, falling back to basic agent: {e}")
            
            # Fallback to basic research agent
            try:
                result = await self.research_agent.conduct_research(query, options.get('context', {}))
                
                # Add citations for sources
                citations = []
                for source in result.sources:
                    citation = self.citation_manager.add_source(
                        url=source.url,
                        title=source.title,
                        content=source.content,
                        metadata={'source_type': source.source_type}
                    )
                    citations.append(citation.id)
                    
                return {
                    'query': result.query.question,
                    'research_type': result.query.research_type.value,
                    'synthesis': result.synthesis,
                    'confidence_level': result.confidence_level,
                    'sources_found': len(result.sources),
                    'gaps_identified': result.gaps_identified,
                    'citations': citations,
                    'timestamp': result.timestamp.isoformat(),
                    'enhanced': False
                }
                
            except Exception as fallback_error:
                self.logger.error(f"Both enhanced and basic research failed: {fallback_error}")
                return {
                    'error': str(fallback_error),
                    'query': query,
                    'timestamp': asyncio.get_event_loop().time(),
                    'enhanced': False
                }
            
    async def search_web(self, query: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search."""
        self.logger.info(f"Searching web for: {query}")
        
        try:
            results = await self.web_search.search(
                query=query,
                num_results=options.get('num_results', 10),
                engines=options.get('engines'),
                scrape_content=options.get('scrape_content', False)
            )
            
            return {
                'query': query,
                'results_found': len(results),
                'results': [
                    {
                        'title': result.title,
                        'url': result.url,
                        'snippet': result.snippet,
                        'source': result.source,
                        'has_content': 'scraped_content' in (result.metadata or {})
                    }
                    for result in results
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {'error': str(e), 'query': query}
            
    def generate_bibliography(self, style: str = 'apa') -> str:
        """Generate bibliography from collected citations."""
        try:
            citation_style = CitationStyle(style.lower())
            return self.citation_manager.generate_bibliography(citation_style)
        except ValueError:
            available_styles = [s.value for s in CitationStyle]
            return f"Invalid citation style. Available styles: {', '.join(available_styles)}"
            
    def export_citations(self, format_type: str = 'json') -> str:
        """Export citations in specified format."""
        try:
            return self.citation_manager.export_citations(format_type)
        except ValueError as e:
            return f"Export failed: {e}"
            
    async def interactive_mode(self):
        """Run in interactive mode."""
        print("Research Agent 2 - Interactive Mode")
        print("Commands: research <query>, search <query>, bibliography, export, quit")
        print("-" * 60)
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if not command:
                    continue
                    
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                elif cmd == 'research' and len(parts) > 1:
                    query = parts[1]
                    result = await self.conduct_research(query, {})
                    self._print_research_result(result)
                elif cmd == 'search' and len(parts) > 1:
                    query = parts[1]
                    result = await self.search_web(query, {'num_results': 5})
                    self._print_search_result(result)
                elif cmd == 'bibliography':
                    bib = self.generate_bibliography()
                    print(bib)
                elif cmd == 'export':
                    citations = self.export_citations('json')
                    print(citations)
                elif cmd == 'help':
                    print("Available commands:")
                    print("  research <query>  - Conduct comprehensive research")
                    print("  search <query>    - Perform web search")
                    print("  bibliography      - Generate bibliography")
                    print("  export            - Export citations as JSON")
                    print("  help              - Show this help")
                    print("  quit              - Exit the program")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    def _print_research_result(self, result: Dict[str, Any]):
        """Print research result in formatted way."""
        if 'error' in result:
            print(f"Research failed: {result['error']}")
            return
            
        print(f"\nResearch Results for: {result['query']}")
        print(f"Enhanced Mode: {'Yes' if result.get('enhanced') else 'No'}")
        
        if result.get('enhanced') and result.get('result'):
            # Enhanced result format
            enhanced_result = result['result']
            print(f"Response Type: {enhanced_result.get('response_type', 'Unknown')}")
            
            if enhanced_result.get('response_type') == 'detailed_analysis':
                methodology = enhanced_result.get('methodology', {})
                print(f"Research Approach: {methodology.get('approach', 'Standard')}")
                
                sources = enhanced_result.get('sources', {})
                print(f"Sources Analyzed: {sources.get('methodology', {}).get('total_sources_reviewed', 0)}")
                
                findings = enhanced_result.get('detailed_findings', {})
                if isinstance(findings, dict) and 'primary_synthesis' in findings:
                    print(f"\nSynthesis:")
                    print(findings['primary_synthesis'])
                    print(f"Confidence: {findings.get('confidence_level', 0):.2f}")
                    
        else:
            # Basic result format
            if 'research_type' in result:
                print(f"Research Type: {result['research_type']}")
            if 'confidence_level' in result:
                print(f"Confidence Level: {result['confidence_level']:.2f}")
            if 'sources_found' in result:
                print(f"Sources Found: {result['sources_found']}")
            if 'synthesis' in result:
                print(f"\nSynthesis:")
                print(result['synthesis'])
            
            if result.get('gaps_identified'):
                print(f"\nInformation Gaps:")
                for gap in result['gaps_identified']:
                    print(f"  - {gap}")
                
    def _print_search_result(self, result: Dict[str, Any]):
        """Print search result in formatted way."""
        if 'error' in result:
            print(f"Search failed: {result['error']}")
            return
            
        print(f"\nSearch Results for: {result['query']}")
        print(f"Results Found: {result['results_found']}")
        
        for i, res in enumerate(result['results'], 1):
            print(f"\n{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Source: {res['source']}")
            if res['snippet']:
                print(f"   Snippet: {res['snippet'][:100]}...")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Research Agent 2 - Intelligent Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive
  python main.py --research "artificial intelligence applications"
  python main.py --search "climate change 2024" --num-results 10
  python main.py --config custom_config.json --research "quantum computing"
        """
    )
    
    # Configuration
    parser.add_argument('--config', '-c', default='config.json',
                      help='Configuration file path (default: config.json)')
    
    # Operation modes
    parser.add_argument('--interactive', '-i', action='store_true',
                      help='Run in interactive mode')
    parser.add_argument('--research', '-r', type=str,
                      help='Conduct research on the given query')
    parser.add_argument('--search', '-s', type=str,
                      help='Perform web search on the given query')
    
    # Options
    parser.add_argument('--num-results', '-n', type=int, default=10,
                      help='Number of search results to retrieve (default: 10)')
    parser.add_argument('--citation-style', default='apa',
                      choices=['apa', 'mla', 'chicago', 'ieee', 'harvard', 'vancouver'],
                      help='Citation style for bibliography (default: apa)')
    parser.add_argument('--output', '-o', type=str,
                      help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Adjust logging level if verbose
    if args.verbose:
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = 'DEBUG'
        
    # Set up logging
    setup_logging(config)
    
    # Create CLI instance
    cli = ResearchAgentCLI(config)
    
    try:
        result = None
        
        if args.interactive:
            await cli.interactive_mode()
        elif args.research:
            result = await cli.conduct_research(args.research, {})
            cli._print_research_result(result)
        elif args.search:
            result = await cli.search_web(args.search, {
                'num_results': args.num_results
            })
            cli._print_search_result(result)
        else:
            print("No operation specified. Use --help for available options.")
            parser.print_help()
            
        # Save output to file if specified
        if args.output and result:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())