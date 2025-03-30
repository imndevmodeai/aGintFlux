import math

def calculator_tool(expression: str) -> str:
    """
    A tool to evaluate mathematical expressions.
    
    Args:
        expression (str): The mathematical expression to evaluate.
        
    Returns:
        str: The result of the evaluation, or an error message.
    """
    try:
        # WARNING: Using eval() is a security risk in real applications
        # This is only used here for simplicity in a controlled environment
        # In a production system, use a safer approach or mathematical parsing library
        result = eval(expression, {"__builtins__": {}}, {
            "math": math,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs
        })
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

if __name__ == "__main__":
    # Example usage
    print(calculator_tool("2 + 3 * 4"))
    print(calculator_tool("math.sqrt(16) + math.sin(math.pi/2)"))
    print(calculator_tool("log10(100) + exp(0)"))
    
    # Error example
    print(calculator_tool("1 / 0")) 