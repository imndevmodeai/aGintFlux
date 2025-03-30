from typing import Dict, Any
import numpy as np

class Tool:
    """Base Tool class for all tools"""
    def __init__(self, name, description, input_schema, output_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        
    def execute(self, input_data):
        raise NotImplementedError("Tool subclasses must implement execute")

class NodeEmbedderTool(Tool):
    """Generates embeddings for knowledge graph nodes"""
    
    def __init__(self):
        super().__init__(
            name="NodeEmbedder",
            description="Generates semantic embeddings for knowledge graph nodes",
            input_schema={"node_text": "str"},
            output_schema={"embedding": "np.ndarray"}
        )
        # We're not using sentence-transformers due to disk space constraints
        
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["node_text"]
        # Simple hash-based embedding generator
        encoding = np.zeros(32, dtype=np.float32)
        for i, char in enumerate(text):
            encoding[i % 32] += ord(char) / 255.0
        
        # Normalize the embedding
        embedding = encoding / np.linalg.norm(encoding)
        return {"embedding": embedding} 