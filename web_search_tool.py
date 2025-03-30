import requests

def web_search_tool(query: str) -> str:
    """
    Performs a web search using a simple request to a search engine and returns the search results.
    For demonstration purposes, this uses a basic, potentially limited, direct request method.
    For production, consider using robust and ethical search APIs (like Google Custom Search API, 
    DuckDuckGo API, or scraping libraries with proper usage policies and rate limiting).

    Args:
        query (str): The search query.

    Returns:
        str: Search results as text, or an error message if the search fails.
    """
    search_url = "https://www.google.com/search"  # Using Google as example
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }  # Adding a user-agent header to mimic browser request

    try:
        response = requests.get(search_url, params={'q': query}, headers=headers, timeout=10) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text  # Return the content as text (HTML) - consider parsing for cleaner results

    except requests.exceptions.RequestException as e:
        return f"Web search failed: {e}"


if __name__ == '__main__':
    search_query = "current weather in London"
    results = web_search_tool(search_query)

    if "failed" not in results:
        print(f"Search results for '{search_query}':\n")
        print(results[:500] + "...") # Print first 500 chars of results
        print("\n[Full HTML output is much longer and would typically be parsed further]")
    else:
        print(f"Error: {results}") 