import requests

def fetch_url_tool(url: str, timeout: int = 30) -> str:
    """
    A tool to fetch content from a URL.
    
    Args:
        url (str): The URL to fetch.
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        
    Returns:
        str: The content of the URL, or an error message.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

if __name__ == "__main__":
    # Example usage
    content = fetch_url_tool("https://example.com")
    
    # Print first 200 characters
    print(content[:200] + "..." if len(content) > 200 else content)
    
    # Error example
    error = fetch_url_tool("https://nonexistent-domain-example.invalid")
    print(error) 