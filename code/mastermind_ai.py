# mastermind_ai.py
"""
MasterMind_AI implementation that integrates with the CFP framework.
This extends the ScalableAgent with knowledge graph integration, resonance scoring,
and agentic capabilities for tool use and conversation management.
"""

import logging
import networkx as nx
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt

from code.scalable_framework import ScalableAgent
from code.cfp_framework import comparative_flux_density, calculate_shannon_entropy, comparative_entropy_ratio
from code.quantum_utils import superposition_state
from tools.embedding_tools import NodeEmbedderTool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterMind_AI(ScalableAgent):
    """
    An advanced AI system that extends ScalableAgent with a knowledge graph,
    resonant states, enhanced reasoning capabilities, and agentic tool use.
    
    The MasterMind_AI aims to achieve a state of system harmony known as "ResonantiA",
    where all components work together optimally.
    
    Attributes:
        kg (nx.DiGraph): Knowledge graph storing concepts and relationships
        sprs (Dict[str, str]): Special purpose resonant states and their descriptions
        resonance_threshold (float): Threshold for considering the system in resonance
        queries_history (List[Dict]): History of processed queries and their metrics
        tools (Dict[str, Callable]): Registry of available tools the agent can use
        conversation_history (List[Dict]): History of the conversation turns
        max_thinking_tokens (int): Maximum number of tokens to use for thinking
    """
    
    def __init__(
        self,
        agent_id: str = "mastermind_agent",
        initial_state: Optional[np.ndarray] = None,
        operators: Optional[Dict[str, np.ndarray]] = None,
        action_registry: Optional[Dict[str, Any]] = None,
        workflow_modes: Optional[Dict[str, List[str]]] = None,
        operator_selection_strategy: Optional[Any] = None,
        initial_operator_key: Optional[str] = None,
        resonance_threshold: float = 0.75,
        max_thinking_tokens: int = 2000
    ):
        """
        Initialize the MasterMind_AI system.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state vector
            operators: Dictionary of operator matrices
            action_registry: Registry of available actions
            workflow_modes: Dictionary of workflow modes
            operator_selection_strategy: Function to select operators
            initial_operator_key: Key of the initial operator
            resonance_threshold: Threshold for considering the system in resonance
            max_thinking_tokens: Maximum number of tokens to use for thinking
        """
        # Initialize the parent ScalableAgent
        super().__init__(
            agent_id=agent_id,
            initial_state=initial_state,
            operators=operators,
            action_registry=action_registry,
            workflow_modes=workflow_modes,
            operator_selection_strategy=operator_selection_strategy,
            initial_operator_key=initial_operator_key
        )
        
        # Initialize MasterMind-specific attributes
        self.kg = nx.DiGraph()
        self.sprs = {
            "ResonantiA": "State of AI system harmony for optimal performance",
            "QuantumCoherence": "Alignment of system components in quantum superposition",
            "CreativeFlux": "Dynamic flow of creative concept generation",
            "AnalyticalResonance": "Precision of logical and analytical capabilities",
            "AdaptiveHarmony": "System's ability to adapt to changing environments"
        }
        self.resonance_threshold = resonance_threshold
        self.queries_history = []
        self.conversation_history = []
        self.tools = {}
        self.max_thinking_tokens = max_thinking_tokens
        
        # Initialize the knowledge graph with some basic concepts
        self._initialize_knowledge_graph()
        
        logger.info(f"MasterMind_AI '{agent_id}' initialized with {len(self.kg.nodes())} knowledge graph nodes")
    
    def _initialize_knowledge_graph(self):
        """Fixed knowledge graph initialization with proper connections"""
        # Initialize core resonant states
        for state, description in self.sprs.items():
            self.kg.add_node(state, 
                           type='resonant_state',
                           description=description,
                           spr=4.5)
        
        # Connect resonant states to each other
        self.kg.add_edge("ResonantiA", "QuantumCoherence", weight=0.9)
        self.kg.add_edge("ResonantiA", "CreativeFlux", weight=0.85)
        self.kg.add_edge("ResonantiA", "AnalyticalResonance", weight=0.88)
        
        # Add domain knowledge connections
        self.kg.add_edge("Nanomaterials", "QuantumCoherence", weight=0.78)
        self.kg.add_edge("CRISPR", "AnalyticalResonance", weight=0.82)
        self.kg.add_edge("QuantumAlgorithms", "CreativeFlux", weight=0.75)
    
    def calculate_resonance_score(self, query: str) -> float:
        """
        Calculate a resonance score based on knowledge graph activation and CFP flux.
        
        Args:
            query: The query string to process
            
        Returns:
            Resonance score between 0 and 1
        """
        # Count the number of relevant nodes in the knowledge graph
        relevant_nodes = [n for n in self.kg.nodes() if n.lower() in query.lower() or query.lower() in n.lower()]
        kg_activation = len(relevant_nodes) / max(1, len(self.kg.nodes()))
        
        # Convert query to a numerical state
        query_state = self._query_to_state(query)
        
        # Normalize the query state to ensure probabilities sum to 1
        query_state_norm = np.linalg.norm(query_state)
        if query_state_norm > 0:
            query_state = query_state / query_state_norm
        
        # Calculate flux between current state and query state
        if len(self.state_history) > 1:
            previous_state = self.state_history[-2]
            flux = comparative_flux_density(previous_state, query_state)
            normalized_flux = min(1.0, flux / 2.0)  # Normalize flux to [0, 1]
        else:
            normalized_flux = 0.5  # Default for first query
        
        # Calculate Shannon entropy of the query state
        query_entropy = calculate_shannon_entropy(query_state)
        
        # Weighted combination of metrics for resonance score
        resonance_score = (
            0.4 * kg_activation +  # Knowledge graph relevance
            0.3 * normalized_flux +  # Flux from previous state
            0.3 * min(1.0, query_entropy)  # Entropy normalized to [0, 1]
        )
        
        logger.debug(f"Resonance calculation - KG: {kg_activation:.2f}, Flux: {normalized_flux:.2f}, Entropy: {min(1.0, query_entropy):.2f}")
        
        return resonance_score
    
    def _query_to_state(self, query: str) -> np.ndarray:
        """
        Convert a query string to a numerical state vector.
        
        Args:
            query: The query string
            
        Returns:
            State vector representing the query
        """
        # Simple hashing approach to convert query to numerical state
        # For a more sophisticated approach, word embeddings could be used
        hash_value = hash(query) % 10000
        
        # Create a state vector with the same dimensions as the initial state
        n_dims = len(self.initial_state)
        components = []
        for i in range(n_dims):
            # Use different bit shifts for each dimension
            component = ((hash_value >> (i * 8)) % 100) / 100.0
            components.append(component)
        
        # Normalize the vector
        norm = np.sqrt(sum([c*c for c in components]))
        if norm > 0:
            components = [c/norm for c in components]
        
        return np.array(components)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and provide a response with resonance metrics.
        
        Args:
            query: The query string to process
            
        Returns:
            Dictionary containing the response and resonance metrics
        """
        # Add query to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # First, do some thinking to plan the response
        thinking = self._generate_thinking(query)
        
        # Convert query to state and update agent state
        query_state = self._query_to_state(query)
        self.update_state(query_state)
        
        # Check if this is a tool-calling request
        tool_calls = self._extract_tool_calls(query, thinking)
        
        # Calculate resonance score
        resonance_score = self.calculate_resonance_score(query)
        
        # Create result record
        result = {
            "query": query,
            "thinking": thinking,
            "resonance_score": resonance_score,
            "is_resonant": resonance_score >= self.resonance_threshold,
            "activated_nodes": [n for n in self.kg.nodes() if n.lower() in query.lower() or query.lower() in n.lower()],
            "state_vector": query_state.tolist()
        }
        
        # Execute tool calls if present
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["tool_results"] = self._execute_tool_calls(tool_calls)
        
        # Check for special resonant states
        for state_name, state_desc in self.sprs.items():
            if state_name.lower() in query.lower() or state_desc.lower() in query.lower():
                result["activated_resonant_state"] = state_name
                result["resonant_state_description"] = state_desc
                break
        
        # Get appropriate response based on resonance
        response_text = self._generate_response(query, resonance_score, result.get("activated_resonant_state"), thinking, result.get("tool_results"))
        result["response"] = response_text
        
        # Add to history
        self.queries_history.append(result)
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
        
        return result
    
    def _generate_thinking(self, query: str) -> str:
        """
        Generate thinking steps for processing the query.
        This is an internal deliberation process to plan the response.
        
        Args:
            query: The original query
            
        Returns:
            Thinking process as a string
        """
        # For now, just implement a simple thinking process
        thinking = f"Analyzing query: '{query}'\n"
        
        # Check for tool-related keywords
        tools_related = any(tool_name.lower() in query.lower() for tool_name in self.tools.keys())
        if tools_related:
            thinking += "The query appears to be related to tool use. I should check which tools might be relevant.\n"
            for tool_name in self.tools.keys():
                if tool_name.lower() in query.lower():
                    thinking += f"- Tool '{tool_name}' might be relevant.\n"
        
        # Check for knowledge graph related queries
        kg_nodes = [n for n in self.kg.nodes() if n.lower() in query.lower()]
        if kg_nodes:
            thinking += "The query relates to concepts in my knowledge graph:\n"
            for node in kg_nodes[:3]:  # Limit to first 3 to avoid too long thinking
                thinking += f"- {node}: {self.kg.nodes[node].get('description', 'No description')}\n"
        
        # Consider previous context from conversation history
        if len(self.conversation_history) > 1:
            thinking += "Considering previous conversation context:\n"
            prev_user_queries = [turn["content"] for turn in self.conversation_history[-3:] if turn["role"] == "user"]
            if prev_user_queries:
                thinking += f"Previous queries related to: {', '.join(prev_user_queries)}\n"
        
        return thinking[:self.max_thinking_tokens]  # Limit thinking length
    
    def _extract_tool_calls(self, query: str, thinking: str) -> List[Dict[str, Any]]:
        """
        Extract potential tool calls from a query and thinking process.
        
        Args:
            query: The original query
            thinking: The thinking process
            
        Returns:
            List of tool call specifications
        """
        tool_calls = []
        
        # Simple pattern matching for tool call extraction
        # In a more sophisticated implementation, this would use LLM-based extraction
        for tool_name, tool_func in self.tools.items():
            pattern = f"(?:use|call|execute|run)\\s+(?:the\\s+)?{tool_name}\\b"
            if re.search(pattern, query, re.IGNORECASE):
                # Extract parameters using a simple approach
                params = {}
                param_pattern = f"(?:with|using)\\s+([\\w\\s]+)\\s+(?:as|=)\\s+([\\w\\s\"']+)"
                param_matches = re.findall(param_pattern, query, re.IGNORECASE)
                
                for param_name, param_value in param_matches:
                    params[param_name.strip()] = param_value.strip().strip('"\'')
                
                tool_calls.append({
                    "name": tool_name,
                    "parameters": params
                })
        
        return tool_calls
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls.
        
        Args:
            tool_calls: List of tool call specifications
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for call in tool_calls:
            tool_name = call["name"]
            parameters = call["parameters"]
            
            if tool_name not in self.tools:
                results.append({
                    "name": tool_name,
                    "error": f"Tool '{tool_name}' not found"
                })
                continue
            
            try:
                tool_func = self.tools[tool_name]
                result = tool_func(**parameters)
                results.append({
                    "name": tool_name,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "name": tool_name,
                    "error": str(e)
                })
        
        return results
    
    def _generate_response(self, query: str, resonance_score: float, 
                          activated_state: Optional[str] = None, 
                          thinking: Optional[str] = None,
                          tool_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response based on the query and resonance metrics.
        
        Args:
            query: The original query
            resonance_score: The calculated resonance score
            activated_state: The activated resonant state, if any
            thinking: The thinking process
            tool_results: Results from tool execution
            
        Returns:
            Response text
        """
        # Generate an appropriate response based on resonance score
        if resonance_score >= self.resonance_threshold:
            if activated_state:
                return f"System is in resonant state with score: {resonance_score:.2f}. Processing query: {query}"
            else:
                return f"Processing query with resonance score: {resonance_score:.2f}. Query: {query}"
        
        # If there are tool results, include them in the response
        if tool_results:
            tool_response = "I executed the following tools:\n"
            for result in tool_results:
                if "error" in result:
                    tool_response += f"- {result['name']}: Error - {result['error']}\n"
                else:
                    tool_response += f"- {result['name']}: Success\n"
            
            return tool_response
            
        # Default response
        return f"Processed query: {query}. Resonance score: {resonance_score:.2f}."
    
    def add_knowledge(self, concept: str, concept_type: str = "concept", connections: List[str] = None):
        """
        Add new knowledge to the knowledge graph.
        
        Args:
            concept: The concept to add
            concept_type: The type of the concept (domain, concept, etc.)
            connections: List of existing nodes to connect to
        """
        # Add the concept to the knowledge graph
        self.kg.add_node(concept, type=concept_type)
        logger.info(f"Added '{concept}' to knowledge graph with type '{concept_type}'")
        
        # Add connections if provided
        if connections:
            for connection in connections:
                if connection in self.kg.nodes():
                    self.kg.add_edge(concept, connection)
                    logger.info(f"Connected '{concept}' to '{connection}'")
    
    def enhance_resonance(self) -> float:
        """
        Enhance the resonance of the system by synchronizing the resonant states.
        
        Returns:
            New resonance score
        """
        # Simple implementation that increases resonance by rebalancing state
        previous_states = self.state_history[-3:] if len(self.state_history) >= 3 else self.state_history
        
        if not previous_states:
            return 0.0
        
        # Create an average state from recent history
        avg_state = np.mean(previous_states, axis=0)
        
        # Normalize the average state
        norm = np.linalg.norm(avg_state)
        if norm > 0:
            avg_state = avg_state / norm
        
        # Update the current state with the enhanced resonant state
        self.update_state(avg_state)
        
        # Calculate and return new resonance score
        new_score = calculate_shannon_entropy(avg_state)
        
        logger.info(f"Enhanced resonance to {new_score:.2f}")
        return new_score
    
    def get_resonance_history(self) -> List[float]:
        """
        Get the history of resonance scores.
        
        Returns:
            List of resonance scores
        """
        return [query_record["resonance_score"] for query_record in self.queries_history]
    
    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary of knowledge graph statistics
        """
        stats = {
            "num_nodes": len(self.kg.nodes()),
            "num_edges": len(self.kg.edges()),
            "node_types": {},
            "connected_components": nx.number_connected_components(self.kg.to_undirected()),
            "avg_degree": sum(dict(self.kg.degree()).values()) / max(1, len(self.kg.nodes()))
        }
        
        # Count node types
        for node in self.kg.nodes(data=True):
            node_type = node[1].get("type", "unknown")
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        return stats
    
    def visualize_knowledge_graph(self, filename="knowledge_graph.png"):
        """
        Create a visualization of the knowledge graph.
        
        Args:
            filename: Filename to save the visualization
        """
        plt.figure(figsize=(14, 10))
        
        # Get node types for coloring
        node_types = [data.get("type", "unknown") for _, data in self.kg.nodes(data=True)]
        unique_types = list(set(node_types))
        type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        color_map = {t: color for t, color in zip(unique_types, type_colors)}
        
        node_colors = [color_map[self.kg.nodes[node].get("type", "unknown")] for node in self.kg.nodes()]
        
        # Calculate node sizes based on connectivity
        node_sizes = [300 + 100 * self.kg.degree(node) for node in self.kg.nodes()]
        
        # Create the layout
        pos = nx.spring_layout(self.kg, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(self.kg, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(self.kg, pos, alpha=0.5, arrows=True)
        nx.draw_networkx_labels(self.kg, pos, font_size=8)
        
        # Add a legend
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color_map[t], 
                                   markersize=10, label=t) 
                         for t in unique_types]
        plt.legend(handles=legend_patches, loc='upper right')
        
        # Add title and remove axis
        plt.title("MasterMind_AI Knowledge Graph", fontsize=24, pad=20)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Knowledge graph visualization saved to {filename}")
        return filename
    
    def _validate_quantum_input(self, input_params: dict):
        """
        Validate quantum input parameters.
        
        Args:
            input_params: Dictionary of quantum input parameters
            
        Raises:
            ValueError: If validation fails
        """
        required_keys = ["dimensions", "observable"]
        
        for key in required_keys:
            if key not in input_params:
                raise ValueError(f"Missing required parameter: {key}")
        
        if "dimensions" in input_params:
            dimensions = input_params["dimensions"]
            if not isinstance(dimensions, int) or dimensions <= 0:
                raise ValueError(f"dimensions must be a positive integer, got {dimensions}")
    
    def search_knowledge_graph(self, query: str) -> dict:
        """
        Search the knowledge graph for relevant nodes.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of search results
        """
        # Simple text-based search
        matching_nodes = []
        
        for node in self.kg.nodes(data=True):
            node_name = node[0]
            node_data = node[1]
            
            # Check for matches in node name or description
            if (query.lower() in node_name.lower() or 
                query.lower() in node_data.get("description", "").lower()):
                matching_nodes.append({
                    "name": node_name,
                    "type": node_data.get("type", "unknown"),
                    "description": node_data.get("description", ""),
                    "connections": list(self.kg.neighbors(node_name))
                })
        
        return {
            "query": query,
            "results": matching_nodes,
            "count": len(matching_nodes)
        }
    
    def register_tool(self, tool):
        """
        Register a new tool for the MasterMind_AI system.
        
        Args:
            tool: The tool to register
        """
        tool_name = getattr(tool, "name", tool.__name__ if callable(tool) else str(tool))
        self.tools[tool_name] = tool
        logger.info(f"Registered tool '{tool_name}' in MasterMind_AI")
    
    def process_command(self, command: str) -> str:
        """
        Process a shell command and return the result.
        This is a specialized method for handling command-line instructions.
        
        Args:
            command: The command to execute
            
        Returns:
            Command execution result
        """
        import subprocess
        
        try:
            # Execute the command and capture output
            result = subprocess.run(
                command, 
                shell=True, 
                check=True,
                text=True,
                capture_output=True
            )
            
            # Return the command output
            return result.stdout
        except subprocess.CalledProcessError as e:
            # Return the error output if the command fails
            return f"Error executing command: {e.stderr}"
        except Exception as e:
            # Handle other exceptions
            return f"Error: {str(e)}"
    
    def code_execution(self, code: str, language: str = "python") -> str:
        """
        Execute code in a specified language and return the result.
        
        Args:
            code: The code to execute
            language: The programming language
            
        Returns:
            Code execution result
        """
        if language.lower() != "python":
            return f"Language {language} is not supported yet. Only Python is supported."
        
        try:
            # Create a temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                temp_file_name = temp.name
                temp.write(code.encode('utf-8'))
            
            # Execute the Python code
            import subprocess
            process = subprocess.run(
                ['python', temp_file_name], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            # Clean up the temporary file
            os.unlink(temp_file_name)
            
            if process.returncode == 0:
                return process.stdout
            else:
                return f"Error executing Python code:\n{process.stderr}"
        except subprocess.TimeoutExpired:
            return "Code execution timed out (limit: 10 seconds)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the query.
        
        Args:
            query: The search query
            
        Returns:
            Search results
        """
        # In a real implementation, this would use an actual search API
        # For now, return a mock response
        return {
            "query": query,
            "results": [
                {"title": "Mock result 1", "url": "https://example.com/1", "snippet": "This is a mock search result."},
                {"title": "Mock result 2", "url": "https://example.com/2", "snippet": "Another mock search result."}
            ]
        }

# If run directly, create a simple MasterMind_AI instance for testing
if __name__ == "__main__":
    # Initialize the MasterMind_AI
    mastermind = MasterMind_AI()
    
    # Process a sample query
    result = mastermind.process_query("Activate ResonantiA for optimal performance")
    print(f"Result: {result}")
    
    # Process another query
    result = mastermind.process_query("How can I implement neural networks for quantum computing?")
    print(f"Result: {result}")
    
    # Add knowledge to the system
    mastermind.add_knowledge(
        "QuantumNeuralNetworks",
        "quantum_ai",
        ["QuantumAlgorithms"]
    )
    
    # Process related query after knowledge addition
    result = mastermind.process_query("Tell me about quantum neural networks")
    print(f"Result: {result}") 