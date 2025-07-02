"""
Research Agent 2 Web Application
Flask-based API server for the web interface
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import uuid
from typing import Dict, Any, Optional

from research_agent import ResearchAgent
from enhanced_research_agent import EnhancedResearchAgent, ResearchRequest
from web_search import EnhancedWebSearch
from api_integrations import create_research_api_manager
from citation_manager import CitationManager, CitationStyle

app = Flask(__name__, static_folder='web_ui/build', static_url_path='')
CORS(app)

# Global instances
research_agent = None
enhanced_agent = None
citation_manager = None
research_sessions = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from file."""
    config_path = 'config_local.json' if os.path.exists('config_local.json') else 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def initialize_agents():
    """Initialize research agents with configuration."""
    global research_agent, enhanced_agent, citation_manager
    
    config = load_config()
    research_agent = ResearchAgent(config.get('research_agent2', {}))
    enhanced_agent = EnhancedResearchAgent(config.get('research_agent2', {}))
    citation_manager = CitationManager()
    
    logger.info("Research agents initialized successfully")


def async_route(f):
    """Decorator to handle async routes in Flask."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapped


@app.route('/')
def index():
    """Serve the React app."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/config')
def get_config():
    """Get configuration status."""
    config = load_config()
    apis_configured = {}
    
    for api_name, api_config in config.get('apis', {}).items():
        auth_config = api_config.get('auth_config', {})
        apis_configured[api_name] = bool(auth_config.get('api_key'))
    
    return jsonify({
        'apis_configured': apis_configured,
        'search_engines': list(config.get('search_engines', {}).keys())
    })


@app.route('/api/research', methods=['POST'])
@async_route
async def conduct_research():
    """Conduct research on a given query."""
    try:
        data = request.json
        query = data.get('query', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        use_enhanced = data.get('use_enhanced', True)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Store session
        research_sessions[session_id] = {
            'query': query,
            'started_at': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        if use_enhanced:
            # Use enhanced agent
            request_obj = ResearchRequest(
                query=query,
                session_id=session_id,
                max_results=data.get('max_results', 10),
                search_engines=data.get('search_engines', ['google', 'bing', 'duckduckgo'])
            )
            
            result = await enhanced_agent.conduct_research(request_obj)
            
            # Convert to JSON-serializable format
            response = {
                'session_id': session_id,
                'query': query,
                'summary': result.get('summary', ''),
                'key_findings': result.get('key_findings', []),
                'sources': [
                    {
                        'url': source.get('url', ''),
                        'title': source.get('title', ''),
                        'snippet': source.get('snippet', ''),
                        'credibility': source.get('credibility', 0)
                    }
                    for source in result.get('sources', [])
                ],
                'confidence_score': result.get('confidence_score', 0),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Use basic agent
            result = await research_agent.conduct_research(query)
            
            response = {
                'session_id': session_id,
                'query': result.query.question,
                'summary': result.synthesis,
                'sources': [
                    {
                        'url': source.url,
                        'title': source.title,
                        'content': source.content[:500] + '...' if len(source.content) > 500 else source.content,
                        'relevance_score': source.relevance_score
                    }
                    for source in result.sources[:10]
                ],
                'confidence_level': result.confidence_level,
                'gaps_identified': result.gaps_identified,
                'timestamp': result.timestamp.isoformat()
            }
        
        # Update session
        research_sessions[session_id]['status'] = 'completed'
        research_sessions[session_id]['completed_at'] = datetime.now().isoformat()
        
        # Add sources to citation manager
        for source in response['sources']:
            citation_manager.add_source(
                url=source.get('url', ''),
                title=source.get('title', ''),
                content=source.get('snippet', source.get('content', ''))
            )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error conducting research: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST'])
@async_route
async def web_search():
    """Perform a web search."""
    try:
        data = request.json
        query = data.get('query', '')
        num_results = data.get('num_results', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        sources = research_agent.search_web(query, num_results)
        
        results = [
            {
                'url': source.url,
                'title': source.title,
                'content': source.content[:300] + '...' if len(source.content) > 300 else source.content,
                'relevance_score': source.relevance_score
            }
            for source in sources
        ]
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/citations', methods=['GET'])
def get_citations():
    """Get current citations in various formats."""
    try:
        style = request.args.get('style', 'apa').upper()
        format_type = request.args.get('format', 'text')
        
        citation_style = getattr(CitationStyle, style, CitationStyle.APA)
        
        if format_type == 'bibtex':
            citations = citation_manager.export_bibtex()
        else:
            citations = citation_manager.generate_bibliography(citation_style)
        
        return jsonify({
            'citations': citations,
            'style': style,
            'format': format_type,
            'count': len(citation_manager.sources)
        })
        
    except Exception as e:
        logger.error(f"Error generating citations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get list of research sessions."""
    return jsonify({
        'sessions': [
            {
                'session_id': sid,
                'query': session['query'],
                'started_at': session['started_at'],
                'status': session['status']
            }
            for sid, session in research_sessions.items()
        ]
    })


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get details of a specific session."""
    session = research_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(session)


@app.route('/api/export/<session_id>', methods=['GET'])
@async_route
async def export_session(session_id):
    """Export session results."""
    try:
        session = research_sessions.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        format_type = request.args.get('format', 'json')
        
        # Get the research result from history
        results = research_agent.get_research_history()
        matching_result = None
        
        for result in results:
            if result.query.question == session['query']:
                matching_result = result
                break
        
        if not matching_result:
            return jsonify({'error': 'No results found for session'}), 404
        
        export_data = research_agent.export_results(matching_result, format_type)
        
        return jsonify({
            'format': format_type,
            'data': export_data
        })
        
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        return jsonify({'error': str(e)}), 500


# Serve React app for all non-API routes
@app.route('/<path:path>')
def serve_react_app(path):
    """Serve React app for client-side routing."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    # Initialize agents on startup
    initialize_agents()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)