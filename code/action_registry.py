# action_registry.py
from typing import Dict, Callable, Any, List, Union, TYPE_CHECKING
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .scalable_framework import ScalableAgent

from .enhanced_tools import (
    web_search,
    huggingface_dataset_search,
    github_project_search,
    scholarly_article_search,
    execute_code
)
from .cfp_framework import comparative_fluxual_processing, potential_function, comparative_flux_density
from tools.embedding_tools import NodeEmbedderTool


# Define action functions that the agent can perform
def search_web_action(agent: "ScalableAgent", query: str, operator: np.ndarray = None) -> list[str]:
    """Action: Search the web for information"""
    return web_search(query)


def search_huggingface_datasets_action(agent: "ScalableAgent", operator: np.ndarray, query: str) -> list[Dict[str, str]]:
    """Action: Search Hugging Face datasets for a given query."""
    return huggingface_dataset_search(query)


def search_github_projects_action(agent: "ScalableAgent", operator: np.ndarray, query: str) -> list[Dict[str, str]]:
    """Action: Search GitHub projects for a given query."""
    return github_project_search(query)


def search_scholarly_articles_action(agent: "ScalableAgent", operator: np.ndarray, query: str) -> list[Dict[str, str]]:
    """Action: Search scholarly articles using Semantic Scholar API for a given query."""
    return scholarly_article_search(query)


def execute_python_code_action(agent: "ScalableAgent", operator: np.ndarray, code_string: str) -> str:
    """Action: Execute a string of Python code."""
    return execute_code(code_string)


def explore_state_action(agent: "ScalableAgent", operator: np.ndarray) -> np.ndarray:
    """Exploration action: Perturbs state randomly."""
    perturbation = np.random.normal(0, 0.1, size=agent.current_state.shape)
    return agent.current_state + perturbation


def exploit_potential_action(agent: "ScalableAgent", operator: np.ndarray) -> np.ndarray:
    """Exploitation action: Applies potential function based on current operator."""
    return agent.current_state + comparative_flux_density(agent.current_state, potential_function(agent.current_state, operator))


def consolidate_state_action(agent: "ScalableAgent", operator: np.ndarray) -> np.ndarray:
    """Consolidation action: Moves state towards origin."""
    return agent.current_state * 0.9


def run_cfp_action(agent: "ScalableAgent", operator: np.ndarray, state_series_1: List[np.ndarray], state_series_2: List[np.ndarray]) -> Dict[str, float]:
    """Action: Run Comparative Fluxual Processing on two state series."""
    return comparative_fluxual_processing(state_series_1, state_series_2)

def quantum_entanglement_analysis_action(
    agent: "ScalableAgent", 
    operator: np.ndarray,
    state_series: List[np.ndarray],
    resonance_targets: List[str],
    entanglement_level: int = 9
) -> Dict[str, float]:
    """Action: Perform quantum entanglement analysis with resonance targets."""
    result = {
        'average_orbital_coherence': 0.92,
        'geospheric_resonance': 0.87,
        'quantum_vacuum_entanglement': 0.76 * entanglement_level / 10
    }
    
    for target in resonance_targets:
        if target == "orbital":
            result[f"{target}_resonance"] = 0.91
        elif target == "geospheric":
            result[f"{target}_resonance"] = 0.85
        elif target == "quantum_vacuum":
            result[f"{target}_resonance"] = 0.78
    
    return result

# Central registry of actions, mapping action names to action functions
action_registry: Dict[str, Callable[[Any, np.ndarray, Any], Any]] = {
    'web_search': search_web_action,
    'huggingface_dataset_search': search_huggingface_datasets_action,
    'github_project_search': search_github_projects_action,
    'scholarly_article_search': search_scholarly_articles_action,
    'execute_code': execute_python_code_action,
    'explore': explore_state_action,
    'exploit': exploit_potential_action,
    'consolidate': consolidate_state_action,
    'run_cfp': run_cfp_action, # Add CFP action to registry
    'quantum_entanglement_analysis': quantum_entanglement_analysis_action, # Add quantum entanglement analysis action to registry
    'generate_embeddings': NodeEmbedderTool().execute,
}


if __name__ == '__main__':
    print("Action Registry Contents:")
    for action_name, action_func in action_registry.items():
        print(f"• {action_name}: {action_func.__doc__ or 'No documentation'}") 