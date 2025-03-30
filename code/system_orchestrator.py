from code.workflow_engine import WorkflowEngine
from code.action_registry import action_registry
from code.quantum_utils import entangled_state, calculate_shannon_entropy
from tools.embedding_tools import NodeEmbedderTool
from spr_tools import CompressSPRsTool, MapSPRToConceptTool
import numpy as np
import networkx as nx
import logging
from typing import Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemOrchestrator:
    """
    Orchestrates the integration between quantum and classical components 
    of the MasterMind_AI system, managing workflows, tools, and system state.
    
    This class handles:
    - Initialization of the MasterMind_AI agent
    - Registration of tools and cognitive workflows
    - Coordination of quantum-classical interfaces
    - Execution of cognitive processes
    """
    
    def __init__(self, mastermind_ai: Optional[Any] = None):
        self.mastermind_ai = mastermind_ai
        self.quantum_system = None
        self.knowledge_graph = None
        
        # Import MasterMind_AI here to avoid circular import
        from code.mastermind_ai import MasterMind_AI
        
        self.mastermind = MasterMind_AI(
            agent_id="quantum_orchestrator",
            initial_state=entangled_state(3),
            action_registry=action_registry,
            resonance_threshold=0.82,
            max_thinking_tokens=2000
        )
        
        self.workflow_engine = WorkflowEngine(self.mastermind, action_registry)
        self.spr_tools = {
            'compress': CompressSPRsTool(),
            'map_concept': MapSPRToConceptTool()
        }
        logger.info("SystemOrchestrator initialized")
        
    def initialize_quantum_system(self):
        """Initialize all quantum-classical interfaces"""
        # Register quantum tools
        for tool_name, tool in self.spr_tools.items():
            self.mastermind.register_tool(tool)
            
        # Link knowledge graph to quantum state
        self.mastermind.kg = self._create_quantum_knowledge_graph()
        
        # Initialize orbital resonance
        self.mastermind.synchronize_orbital_resonance()
        logger.info(f"Quantum system initialized with {len(self.mastermind.kg.nodes())} knowledge graph nodes")
        
    def execute_cognitive_workflow(self, query: str) -> dict:
        """
        Execute the full cognitive processing pipeline.
        
        Args:
            query: The input query string to process
            
        Returns:
            Dict containing the processing results, including response and metrics
        """
        # Process the query using the MasterMind_AI's process_query method
        result = self.mastermind.process_query(query)
        
        # Add additional quantum metrics
        result["quantum_metrics"] = self._calculate_quantum_metrics()
        
        # Map to concept if not already done
        if "concept_map" not in result:
            # Generate simple SPR from query
            sprs = []
            for i in range(3):
                # Create a simple representation based on character encoding
                query_chars = query[i:i+10] if i < len(query) else query[:10]  # Get 10 chars of query with offset
                # Convert characters to vector
                query_vec = np.zeros(8)
                for j, char in enumerate(query_chars[:8]):
                    query_vec[j % 8] = ord(char) / 255.0
                # Normalize
                if np.linalg.norm(query_vec) > 0:
                    query_vec = query_vec / np.linalg.norm(query_vec)
                sprs.append(query_vec)
            
            # Compress SPRs
            spr = self.spr_tools['compress'].execute({
                "sprs": sprs,
                "weights": [0.6, 0.3, 0.1]
            })['compressed_spr']
            
            # Map to concept
            concept_map = self.spr_tools['map_concept'].execute({
                "spr": spr,
                "knowledge_graph": self.mastermind.kg
            })
            
            result["concept_map"] = {
                "concept": concept_map.get("concept", "Unknown"),
                "confidence": concept_map.get("confidence", 0.0)
            }
        
        # Log the processing metrics
        logger.info(f"Query processed with resonance: {result['resonance_score']:.2f}")
        
        return result

    def _create_quantum_knowledge_graph(self) -> nx.DiGraph:
        """Initialize quantum-enhanced knowledge graph"""
        kg = nx.DiGraph()
        
        # Add core quantum concepts
        kg.add_node("QuantumSuperposition", 
                   embedding=NodeEmbedderTool().execute({"node_text": "QuantumSuperposition"})['embedding'],
                   type='quantum',
                   spr=4.25)
        
        # Add nanotech domain
        kg.add_node("Nanomaterials",
                   embedding=NodeEmbedderTool().execute({"node_text": "Nanomaterials"})['embedding'],
                   type='domain',
                   spr=4.31)
        
        kg.add_edge("QuantumSuperposition", "Nanomaterials", relation="enables")
        
        # Add the resonant states from MasterMind_AI
        for state, description in self.mastermind.sprs.items():
            kg.add_node(state,
                      embedding=NodeEmbedderTool().execute({"node_text": f"{state}: {description}"})['embedding'],
                      type='resonant_state',
                      description=description,
                      spr=4.5)
        
        return kg

    def _calculate_quantum_metrics(self) -> dict:
        """Calculate system-wide quantum metrics"""
        # Initialize quantum state if not already present
        if self.mastermind.quantum_state is None:
            self.mastermind.quantum_state = np.array([0.5, 0.5, 0.5, 0.5])
        
        return {
            "coherence": float(np.mean(self.mastermind.quantum_state)),
            "entanglement": float(calculate_shannon_entropy(self.mastermind.quantum_state)),
            "resonance": float(self.mastermind.orbital_sync_factor)
        }
        
    def run_agentic_workflow(self, query: str, tool_names: list = None) -> dict:
        """
        Run a workflow that makes use of agentic capabilities.
        
        Args:
            query: The query or instruction to process
            tool_names: Optional list of specific tools to use
            
        Returns:
            Dict containing the workflow results
        """
        logger.info(f"Running agentic workflow with query: {query}")
        
        # Process the query using the mastermind's process_query method
        # This will automatically handle tool calls based on the query
        result = self.mastermind.process_query(query)
        
        # If specific tools were requested but not used, try to use them explicitly
        if tool_names and not result.get('tool_calls'):
            logger.info(f"No automatic tool calls detected, trying requested tools: {tool_names}")
            
            tool_calls = []
            for tool_name in tool_names:
                if tool_name in self.mastermind.tools:
                    tool_calls.append({
                        "name": tool_name,
                        "parameters": {"query": query}
                    })
            
            if tool_calls:
                # Execute the tool calls
                tool_results = self.mastermind._execute_tool_calls(tool_calls)
                result["tool_calls"] = tool_calls
                result["tool_results"] = tool_results
                
                # Generate a new response that includes the tool results
                response_text = self.mastermind._generate_response(
                    query, 
                    result["resonance_score"],
                    result.get("activated_resonant_state"),
                    result.get("thinking"),
                    tool_results
                )
                result["response"] = response_text
                
                # Update the conversation history
                self.mastermind.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
        
        # Add quantum metrics
        result["quantum_metrics"] = self._calculate_quantum_metrics()
        
        return result 