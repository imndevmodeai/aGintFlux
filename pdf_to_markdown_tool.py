def pdf_to_markdown_tool(pdf_path: str) -> str:
    """
    A tool to convert PDF documents to Markdown format.
    This is a stub implementation that returns a placeholder message.
    
    Args:
        pdf_path (str): The path to the PDF document.
        
    Returns:
        str: A placeholder message indicating PDF to Markdown conversion.
    """
    import os
    
    file_extension = os.path.splitext(pdf_path)[1].lower()
    
    if file_extension != '.pdf':
        return f"Error: Expected a PDF file, but got {file_extension} file: {pdf_path}"
    
    return f"PDF to Markdown conversion (stub): {pdf_path}\n"\
           f"In a complete implementation, this would convert the PDF to Markdown format."

if __name__ == "__main__":
    # Example usage
    result = pdf_to_markdown_tool("document.pdf")
    print(result)
    
    # Example with non-PDF file
    result = pdf_to_markdown_tool("text.txt")
    print(result) 