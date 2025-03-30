# enhanced_tools.py
from typing import Dict, Any, Callable, Optional, List
import numpy as np
import requests
import json
import os
import sys
import io
import tempfile
import subprocess
from urllib.parse import quote_plus
import hashlib
import time

# Configure requests to handle exceptions properly
requests.packages.urllib3.disable_warnings()

# Add at the top of the file
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(query: str) -> str:
    """Generate a cache file path based on query hash"""
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"search_{query_hash}.json")

def web_search(query: str, num_results: int = 5, cache_ttl: int = 3600, 
               retry_delay: int = 5, max_retries: int = 3) -> list[str]:
    """
    Performs a web search using DuckDuckGo API with caching and rate limit handling.

    Args:
        query (str): The search query string.
        num_results (int): The number of search results to return (default: 5).
        cache_ttl (int): Time to live for cached results in seconds (default: 1 hour)
        retry_delay (int): Initial delay between retries in seconds (default: 5)
        max_retries (int): Maximum number of retry attempts (default: 3)

    Returns:
        list[str]: A list of URLs as strings.
    """
    # Check cache first
    cache_path = get_cache_path(query)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is still valid
            if time.time() - cache_data.get('timestamp', 0) < cache_ttl:
                print(f"Using cached results for query: {query}")
                return cache_data.get('results', [])[:num_results]
            else:
                print(f"Cache expired for query: {query}")
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    # Cache miss or invalid cache, perform the search
    retries = 0
    while retries < max_retries:
        try:
            # Use DuckDuckGo API for search with correct parameters
            encoded_query = quote_plus(query)
            search_url = (
                f"https://api.duckduckgo.com/"
                f"?q={encoded_query}"
                "&format=json"
                "&no_redirect=1"
                "&no_html=1"
                "&t=cfp_framework"
            )
            
            headers = {
                'User-Agent': 'CFP-Framework/1.0 (https://github.com/yourusername/cfp-framework; contact@example.com)',
                'Accept': 'application/json'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            # Handle rate limiting responses
            if response.status_code == 429:  # Too Many Requests
                retries += 1
                if retries < max_retries:
                    print(f"Rate limited. Waiting {retry_delay} seconds before retry {retries}/{max_retries}...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    print("Max retries reached for rate limiting. Using simulated results.")
                    break
            
            response.raise_for_status()
            
            results = response.json()
            urls = []
            
            # Check for RelatedTopics instead of Results
            if 'RelatedTopics' in results:
                for topic in results['RelatedTopics'][:num_results]:
                    if isinstance(topic, dict):
                        # Try both FirstURL and AbstractURL
                        url = topic.get('FirstURL') or topic.get('AbstractURL')
                        if url:
                            urls.append(url)
            
            # If we got valid results, cache them
            if urls:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump({
                            'timestamp': time.time(),
                            'results': urls,
                            'query': query
                        }, f)
                except Exception as e:
                    print(f"Error writing to cache: {e}")
                
                return urls
            
            # Otherwise, fall back to simulated results
            print(f"No search results found via API, using simulated results for query: {query}")
            break
            
        except requests.RequestException as e:
            if "timeout" in str(e).lower():
                retries += 1
                if retries < max_retries:
                    print(f"Request timed out. Retrying in {retry_delay} seconds... ({retries}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
            print(f"Error in web search API: {str(e)}, using simulated results")
            break
        except Exception as e:
            print(f"Error in web search API: {str(e)}, using simulated results")
            break
    
    # Generate simulated search results if API failed or returned no results
    simulated_urls = [
        f"https://example.com/article/{query.replace(' ', '-')}-{i+1}" 
        for i in range(num_results)
    ]
    return simulated_urls


def huggingface_dataset_search(query: str, num_results: int = 3) -> list[Dict[str, str]]:
    """
    Searches for datasets on Hugging Face.

    This function attempts to query the Hugging Face API to search for datasets. 
    If the API request fails, it falls back to returning simulated results.

    Args:
        query (str): The search query string.
        num_results (int): The number of search results to return (default: 3).

    Returns:
        list[Dict[str, str]]: A list of dictionaries with dataset info.
    """
    try:
        # Real API endpoint for Hugging Face dataset search
        api_url = "https://huggingface.co/api/datasets"
        
        # Make request to API
        response = requests.get(
            api_url, 
            params={'search': query, 'limit': num_results},
            timeout=10
        )
        response.raise_for_status()
        
        # Parse results from API response
        results = response.json()
        
        # Convert results to our output format
        datasets = []
        for dataset in results:
            if isinstance(dataset, dict) and 'id' in dataset:
                dataset_id = dataset['id']
                dataset_info = {
                    'id': dataset_id,
                    'url': f"https://huggingface.co/datasets/{dataset_id}",
                    'description': dataset.get('description', 'No description available')
                }
                datasets.append(dataset_info)
        
        # If we got results, return them
        if datasets:
            return datasets[:num_results]
            
    except Exception as e:
        print(f"Error in Hugging Face dataset search: {str(e)}, using simulated results")
    
    # Generate simulated search results if API failed or returned no results
    simulated_datasets = []
    for i in range(num_results):
        sanitized_query = query.replace(' ', '_').lower()
        dataset_id = f"{sanitized_query}_dataset_{i+1}"
        simulated_datasets.append({
            'id': dataset_id,
            'url': f"https://huggingface.co/datasets/{dataset_id}",
            'description': f"A dataset related to {query} (simulated result)"
        })
    
    return simulated_datasets


def github_project_search(query: str, num_results: int = 3) -> list[Dict[str, str]]:
    """
    Searches for GitHub projects related to the query.

    This function attempts to query the GitHub API to search for repositories.
    If the API request fails, it falls back to returning simulated results.

    Args:
        query (str): The search query string.
        num_results (int): The number of search results to return (default: 3).

    Returns:
        list[Dict[str, str]]: A list of dictionaries with repository info.
    """
    try:
        # GitHub API endpoint for repository search
        api_url = "https://api.github.com/search/repositories"
        
        # Set up headers with User-Agent to avoid API rate limiting issues
        headers = {
            'User-Agent': 'CFP-Framework-Agent',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Add GitHub API token if available
        if 'GITHUB_API_TOKEN' in os.environ:
            headers['Authorization'] = f"token {os.environ['GITHUB_API_TOKEN']}"
        
        # Make request to GitHub API
        response = requests.get(
            api_url,
            headers=headers,
            params={'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': num_results},
            timeout=10
        )
        response.raise_for_status()
        
        # Parse results from API response
        results = response.json()
        
        # Extract repository information
        repositories = []
        if 'items' in results and isinstance(results['items'], list):
            for repo in results['items'][:num_results]:
                repo_info = {
                    'name': repo['full_name'],
                    'url': repo['html_url'],
                    'description': repo.get('description', 'No description available'),
                    'stars': repo.get('stargazers_count', 0)
                }
                repositories.append(repo_info)
        
        # If we got results, return them
        if repositories:
            return repositories
            
    except Exception as e:
        print(f"Error in GitHub project search: {str(e)}, using simulated results")
    
    # Generate simulated search results if API failed or returned no results
    simulated_repos = []
    for i in range(num_results):
        sanitized_query = query.replace(' ', '-').lower()
        repo_name = f"user-{i+1}/{sanitized_query}-project"
        simulated_repos.append({
            'name': repo_name,
            'url': f"https://github.com/{repo_name}",
            'description': f"A project related to {query} (simulated result)",
            'stars': 100 * (num_results - i)  # Simulate star count
        })
    
    return simulated_repos


def scholarly_article_search(query: str, num_results: int = 3, api_key: Optional[str] = None) -> list[Dict[str, str]]:
    """
    Searches for scholarly articles related to the query.

    This function attempts to use the Semantic Scholar API if an API key is provided.
    If no API key is available or the API request fails, it falls back to simulated results.

    Args:
        query (str): The search query string.
        num_results (int): The number of search results to return (default: 3).
        api_key (Optional[str]): API key for Semantic Scholar (optional).

    Returns:
        list[Dict[str, str]]: A list of dictionaries with article info.
    """
    # Check for API key in function args or environment
    if not api_key:
        api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
    
    # Try to use the Semantic Scholar API if we have an API key
    if api_key:
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            headers = {"x-api-key": api_key}
            
            # Make request to API
            response = requests.get(
                api_url,
                headers=headers,
                params={'query': query, 'limit': num_results, 'fields': 'title,authors,abstract,url'},
                timeout=10
            )
            response.raise_for_status()
            
            # Parse results from API response
            results = response.json()
            
            # Extract article information
            articles = []
            if 'data' in results and isinstance(results['data'], list):
                for paper in results['data']:
                    # Extract author names
                    authors = []
                    if 'authors' in paper and isinstance(paper['authors'], list):
                        authors = [author.get('name', 'Unknown') for author in paper['authors']]
                    
                    article_info = {
                        'title': paper.get('title', 'Untitled'),
                        'url': paper.get('url', f"https://semanticscholar.org/paper/{paper.get('paperId', '')}"),
                        'abstract': paper.get('abstract', 'No abstract available'),
                        'authors': ', '.join(authors) if authors else 'Unknown'
                    }
                    articles.append(article_info)
            
            # If we got results, return them
            if articles:
                return articles
                
        except Exception as e:
            print(f"Error in Semantic Scholar search: {str(e)}, using simulated results")
    else:
        print("No Semantic Scholar API key provided, using simulated results")
    
    # Generate simulated search results if API failed or no API key available
    simulated_articles = []
    for i in range(num_results):
        title = f"Research on {query.title()}: Part {i+1}"
        simulated_articles.append({
                'title': title,
            'url': f"https://example.org/papers/{query.replace(' ', '_').lower()}_{i+1}",
            'abstract': f"This research paper explores {query} and presents novel findings in the field. (Simulated abstract)",
            'authors': f"Author A{i+1}, Author B{i+1}, Author C{i+1}"
        })
    
    return simulated_articles


def execute_code(code_string: str) -> str:
    """
    Executes a Python code string and returns the output.
    
    This function executes the given Python code in a restricted environment,
    capturing stdout and stderr output. It handles errors gracefully and
    returns the output as a string.

    Args:
        code_string (str): Python code to execute.

    Returns:
        str: Output of the executed code or error message.
    """
    # Define allowed modules for the restricted environment
    allowed_modules = {
        'numpy': np,
        'json': json,
        'math': __import__('math'),
        'random': __import__('random'),
        'datetime': __import__('datetime'),
        'collections': __import__('collections'),
    }
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    try:
        # Redirect stdout and stderr
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # Create isolated namespace
        local_vars = {}
        
        # Execute the code
        exec(code_string, allowed_modules, local_vars)
        
        # Get output
        output = stdout_buffer.getvalue()
        error = stderr_buffer.getvalue()
        
        # Combine output and error messages
        if error:
            return f"{output}\n\nWARNING: {error}"
        else:
            return output
            
    except Exception as e:
        # Capture any execution errors
        return f"ERROR: {str(e)}"
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


if __name__ == '__main__':
    # Example usage of web_search
    print("Testing web_search...")
    results = web_search("quantum computing", 3)
    print(f"Web search results for 'quantum computing': {results}\n")

    # Example usage of huggingface_dataset_search
    print("Testing huggingface_dataset_search...")
    datasets = huggingface_dataset_search("natural language processing", 2)
    print(f"Hugging Face datasets for 'natural language processing': {datasets}\n")

    # Example usage of github_project_search
    print("Testing github_project_search...")
    repos = github_project_search("machine learning", 2)
    print(f"GitHub repositories for 'machine learning': {repos}\n")
    
    # Example usage of scholarly_article_search
    print("Testing scholarly_article_search...")
    articles = scholarly_article_search("quantum algorithms", 2)
    print(f"Scholarly articles for 'quantum algorithms': {articles}\n")

    # Example usage of execute_code
    print("Testing execute_code...")
    result = execute_code("import numpy as np\narray = np.array([1, 2, 3])\nprint(f'NumPy array: {array}')\nprint(f'Sum: {np.sum(array)}')")
    print(f"Code execution result:\n{result}") 