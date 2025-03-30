import subprocess
import tempfile
import os

def code_execution_tool(code: str, language: str = "python") -> str:
    """
    Executes code in a specified language and returns the output.
    Currently supports Python, with stub implementations for other languages.

    Args:
        code (str): The code to execute.
        language (str, optional): The programming language. Defaults to "python".

    Returns:
        str: The output of the code execution, or an error message if execution fails.
    """

    if language.lower() == "python":
        return _execute_python(code)
    else:
        return f"Language '{language}' is not supported in this implementation."


def _execute_python(code: str) -> str:
    """Executes Python code in a sandbox environment."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(code)
            temp_file_name = tmp_file.name

        process = subprocess.run(['python', temp_file_name], capture_output=True, text=True, timeout=10)
        os.remove(temp_file_name)  # Clean up temporary file
        
        if process.returncode == 0:
            return process.stdout
        else:
            return f"Error executing Python code:\n{process.stderr}"

    except subprocess.TimeoutExpired:
        os.remove(temp_file_name)  # Clean up temporary file
        return "Error: Code execution timed out (exceeded 10 seconds)"
    except Exception as e:
        try:
            os.remove(temp_file_name)  # Attempt to clean up temporary file
        except:
            pass  # Ignore if file already removed or can't be removed
        return f"Error executing Python code: {e}"


if __name__ == '__main__':
    # Example Usage
    python_code = "print('Hello, World!')\nprint(2 + 2)"
    result = code_execution_tool(python_code)
    print(f"Execution Result:\n{result}")
    
    # Example with an error
    error_code = "print(undefined_variable)"
    error_result = code_execution_tool(error_code)
    print(f"\nError Execution Result:\n{error_result}")
    
    # Example with timeout
    timeout_code = "import time\nwhile True: time.sleep(1)"
    timeout_result = code_execution_tool(timeout_code)
    print(f"\nTimeout Execution Result:\n{timeout_result}") 