import numpy as np
from typing import Dict, List, Any, Optional

class Tool:
    """Base Tool class for all tools"""
    def __init__(self, name, description, input_schema, output_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        
    def execute(self, input_data):
        raise NotImplementedError("Tool subclasses must implement execute")

def create_sparse_priming_representations(s_system, parameters, num_sprs=5):
    """
    Creates multiple Sparse Priming Representations (SPRs) using S-system models.
    
    Args:
        s_system (np.ndarray): A 2x2 matrix representing the S-system.
        parameters (np.ndarray): Parameters for controlling the SPR generation.
        num_sprs (int): Number of SPRs to generate.
        
    Returns:
        List[np.ndarray]: A list of SPR vectors.
    """
    sprs = []
    
    # Generate multiple SPRs
    for i in range(num_sprs):
        # Add some randomness to the parameters for variation
        perturbed_params = parameters + np.random.normal(0, 0.1, size=parameters.shape)
        
        # Simple S-system equation: dx/dt = alpha * x^g - beta * x^h
        # Here we simplify by using matrix multiplication
        spr = s_system @ perturbed_params
        
        # Normalize the SPR
        spr = spr / np.linalg.norm(spr)
        
        sprs.append(spr)
    
    return sprs

class CompressSPRsTool(Tool):
    """Compresses multiple SPRs into a single representation."""
    
    def __init__(self):
        super().__init__(
            name="CompressSPRs",
            description="Compresses multiple SPRs into a single representation using weighted combination.",
            input_schema={
                "sprs": "List[np.ndarray]",
                "weights": "Optional[List[float]]"
            },
            output_schema={"compressed_spr": "np.ndarray"}
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        sprs = input_data["sprs"]
        weights = input_data.get("weights")
        
        if not weights:
            # Equal weighting if no weights provided
            weights = [1.0/len(sprs)] * len(sprs)
        
        # Ensure weights sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Weighted combination
        compressed_spr = np.zeros_like(sprs[0])
        for spr, weight in zip(sprs, weights):
            compressed_spr += weight * spr
        
        # Normalize
        compressed_spr = compressed_spr / np.linalg.norm(compressed_spr)
        
        return {"compressed_spr": compressed_spr}

class CalculateSPRSimilarityTool(Tool):
    """Calculates similarity between SPRs."""
    
    def __init__(self):
        super().__init__(
            name="CalculateSPRSimilarity",
            description="Calculates similarity between two SPRs using cosine similarity.",
            input_schema={
                "spr_a": "np.ndarray",
                "spr_b": "np.ndarray"
            },
            output_schema={"similarity": "float"}
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        spr_a = input_data["spr_a"]
        spr_b = input_data["spr_b"]
        
        # Calculate cosine similarity
        similarity = np.dot(spr_a, spr_b) / (np.linalg.norm(spr_a) * np.linalg.norm(spr_b))
        
        return {"similarity": float(similarity)}

class MapSPRToConceptTool(Tool):
    """Maps an SPR to a concept in a knowledge graph."""
    
    def __init__(self):
        super().__init__(
            name="MapSPRToConcept",
            description="Maps an SPR to a concept in the knowledge graph.",
            input_schema={
                "spr": "np.ndarray",
                "knowledge_graph": "Dict[str, Any]"
            },
            output_schema={
                "concept": "str",
                "confidence": "float"
            }
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        spr = input_data["spr"]
        knowledge_graph = input_data["knowledge_graph"]
        
        # Get all node embeddings
        node_embeddings = {
            node: knowledge_graph.nodes[node].get('embedding', np.zeros(32))  # Changed to match SPR dimension
            for node in knowledge_graph.nodes
        }
        
        # Calculate cosine similarities
        similarities = {}
        spr_norm = np.linalg.norm(spr)
        
        # Ensure SPR has dimension 32 to match embeddings
        if spr.shape[0] != 32:
            # Resize SPR by padding or truncating
            resized_spr = np.zeros(32)
            min_dim = min(spr.shape[0], 32)
            resized_spr[:min_dim] = spr[:min_dim]
            spr = resized_spr
            spr_norm = np.linalg.norm(spr)
        
        for node, emb in node_embeddings.items():
            # Ensure embedding has dimension 32
            if emb.shape[0] != 32:
                resized_emb = np.zeros(32)
                min_dim = min(emb.shape[0], 32)
                resized_emb[:min_dim] = emb[:min_dim]
                emb = resized_emb
            
            emb_norm = np.linalg.norm(emb)
            if spr_norm == 0 or emb_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(spr, emb) / (spr_norm * emb_norm)
            similarities[node] = similarity
        
        # Find best match
        concept = max(similarities, key=similarities.get) if similarities else "Unknown"
        confidence = similarities.get(concept, 0.0)
        return {"concept": concept, "confidence": confidence}

class ExtractSPRFeaturesTool(Tool):
    """Extracts features from an SPR representation."""
    
    def __init__(self):
        super().__init__(
            name="ExtractSPRFeatures",
            description="Extracts key features from an SPR representation for analysis.",
            input_schema={"spr": "np.ndarray"},
            output_schema={"features": "Dict[str, float]"}
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        spr = input_data["spr"]
        
        # Calculate various statistical properties of the SPR
        features = {
            "mean": float(np.mean(spr)),
            "std_dev": float(np.std(spr)),
            "max_value": float(np.max(spr)),
            "min_value": float(np.min(spr)),
            "l2_norm": float(np.linalg.norm(spr)),
            "entropy": float(-np.sum(np.abs(spr) * np.log(np.abs(spr) + 1e-10)))
        }
        
        return {"features": features} 