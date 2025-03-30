# workflow_engine.py
from typing import Dict, List, Callable, Any, Optional
import json
import os  # Add import for os module

import numpy as np

from code.scalable_framework import ScalableAgent
from code.enhanced_tools import (
    web_search,
    huggingface_dataset_search,
    github_project_search,
    scholarly_article_search,
    execute_code
)
from code.cfp_framework import comparative_fluxual_processing
from code.quantum_utils import quantum_state_evolution



class WorkflowEngine:
    """
    A workflow engine to manage and execute complex workflows defined in JSON format,
    utilizing a ScalableAgent and a registry of actions.

    This engine parses a workflow definition from a JSON file, which specifies a sequence of
    actions, parameters, and control flow logic. It executes these workflows using a
    ScalableAgent, allowing for dynamic and flexible task automation.

    Attributes:
        agent (ScalableAgent): The agent responsible for executing actions.
        action_registry (Dict[str, Callable]): Registry of available actions (functions).

    Args:
        agent (ScalableAgent): ScalableAgent instance to be used for workflow execution.
        action_registry (Dict[str, Callable]): Action registry, mapping action names to functions.

    Example Workflow JSON (cfp_workflow.json):
    [
      {"action": "web_search", "params": {"query": "quantum cryptography"}},
      {"action": "execute_code", "params": {"code_string": "print(2+2)"}},
      {"action": "run_cfp",
       "params": {
           "state_series_1": "get_agent_state_history",
           "state_series_2": "another_agent.get_state_history"
        },
       "context": {"another_agent": "agent_instance_2"}
      }
    ]
    """
    def __init__(self, agent: ScalableAgent, action_registry: Dict[str, Callable]):
        self.agent = agent
        self.action_registry = action_registry
        self.workflow_results = {} # Store results of workflow executions


    def execute_workflow_from_json(self, workflow_json_path: str, workflow_id: str) -> List[Any]:
        """
        Executes a workflow defined in a JSON file.

        Parses the JSON workflow definition, iterates through each step, and executes
        the corresponding action using the ScalableAgent. Supports action parameters and
        context variables within the workflow definition.

        Args:
            workflow_json_path (str): Path to the JSON file defining the workflow.
            workflow_id (str): Unique identifier for this workflow execution.

        Returns:
            List[Any]: List of results from each step in the workflow.

        Raises:
            FileNotFoundError: if workflow_json_path does not exist.
            json.JSONDecodeError: if workflow_json_path is not valid JSON.
            ValueError: if 'action' key is missing in any workflow step definition.
            ValueError: if action name in workflow step is not found in action_registry.
        """
        try:
            with open(workflow_json_path, 'r') as f: # Open and read JSON workflow file
                workflow_definition = json.load(f) # Load JSON data
        except FileNotFoundError:
            raise FileNotFoundError(f"Workflow JSON file not found: {workflow_json_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in workflow file: {workflow_json_path} - {e.msg}", e.doc, e.pos)


        if not isinstance(workflow_definition, list): # Workflow definition should be a list of steps
            raise ValueError("Workflow definition must be a JSON list of workflow steps.")


        workflow_step_results = [] # Store results for each step in the workflow
        context_variables = {'agent': self.agent} # Initialize context with the agent instance


        for step_definition in workflow_definition: # Iterate through each step in the workflow
            if 'action' not in step_definition: # Each step must define an 'action'
                raise ValueError("Workflow step definition missing 'action' key.")

            action_name = step_definition['action'] # Get action name from step definition
            params = step_definition.get('params', {}) # Get parameters for the action, default to empty dict if not provided
            context_update = step_definition.get('context', {}) # Get context updates for the step, default to empty dict
            step_mode = step_definition.get('mode', 'default') # Get step mode, default to 'default'

            if action_name not in self.action_registry: # Validate action name against action registry
                raise ValueError(f"Action '{action_name}' not found in action registry.")


            # Resolve dynamic parameters and context variables before action execution
            resolved_params = self._resolve_parameters(params, context_variables)
            context_variables.update(context_update) # Update context with step-specific context

            try:
                if step_mode == 'run_cfp' and action_name == 'run_cfp':
                    # Handle CFP action specifically, resolving state series from context
                    cfp_results = self._execute_cfp_action(self.action_registry[action_name], resolved_params, context_variables)
                    step_result = cfp_results # Step result is CFP metrics dictionary
                else:
                    # Use the agent's perform_action method directly instead of calling the function
                    step_result = self.agent.perform_action(action_name, **resolved_params)
                
                workflow_step_results.append(step_result) # Store result of the step

            except Exception as e: # Catch exceptions during action execution
                error_msg = f"Error executing action '{action_name}' in workflow '{workflow_id}', step {workflow_definition.index(step_definition) + 1}: {e}"
                print(error_msg) # Print error message to console
                workflow_step_results.append({'error': error_msg}) # Append error info to results


        self.workflow_results[workflow_id] = workflow_step_results # Store all step results in workflow_results
        return workflow_step_results # Return list of step results for the workflow


    def _resolve_parameters(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolves dynamic parameters within a workflow step definition.

        Checks if parameter values are strings referring to context variables or agent methods,
        and resolves them to their actual values before action execution.

        Args:
            params (Dict[str, Any]): Parameters dictionary from workflow step definition.
            context (Dict[str, Any]): Current context variables dictionary.

        Returns:
            Dict[str, Any]: Resolved parameters dictionary with dynamic references replaced by actual values.
        """
        resolved_params = {}
        for key, value in params.items(): # Iterate through parameters
            if isinstance(value, str) and value.startswith('get_agent_'): # Check for dynamic parameter referencing agent method
                method_name = value[len('get_agent_'):] # Extract method name
                if hasattr(self.agent, method_name) and callable(getattr(self.agent, method_name)): # Verify method existence and callable
                    resolved_params[key] = getattr(self.agent, method_name)() # Call agent method and set as parameter value
                elif value == 'get_agent_state_history': # Special case for accessing state history directly
                    resolved_params[key] = self.agent.state_history # Get state history
                else:
                    resolved_params[key] = value # If not resolvable, keep original string value (or handle differently)
            elif isinstance(value, str) and value in context: # Check if parameter is context variable
                resolved_params[key] = context[value] # Resolve to context variable value
            else:
                resolved_params[key] = value # Use parameter value directly if not dynamic reference or context variable

        return resolved_params # Return dictionary with resolved parameter values


    def _execute_cfp_action(self, action_function: Callable, params: Dict[str, Any], context_variables: Dict[str, Any]) -> Dict[str, float]:
        """
        Executes the 'run_cfp' action, specifically handling state series parameters.

        Resolves state series parameters that might be strings referring to agent state histories
        or context variables, and then executes the comparative_fluxual_processing function.

        Args:
            action_function (Callable): The CFP action function (comparative_fluxual_processing).
            params (Dict[str, Any]): Parameters dictionary for the CFP action.
            context_variables (Dict[str, Any]): Current context variables dictionary.

        Returns:
            Dict[str, float]: Results from the comparative_fluxual_processing function (CFP metrics).

        Raises:
            ValueError: if state_series_1 or state_series_2 parameters are not resolved to lists of numpy arrays.
        """
        state_series_1_param = params.get('state_series_1') # Get state_series_1 parameter
        state_series_2_param = params.get('state_series_2') # Get state_series_2 parameter

        state_series_1 = self._resolve_state_series_parameter(state_series_1_param, context_variables) # Resolve state_series_1
        state_series_2 = self._resolve_state_series_parameter(state_series_2_param, context_variables) # Resolve state_series_2


        if not isinstance(state_series_1, list) or not all(isinstance(state, np.ndarray) for state in state_series_1): # Validate series_1
            raise ValueError("Parameter 'state_series_1' must resolve to a list of numpy arrays.")
        if not isinstance(state_series_2, list) or not all(isinstance(state, np.ndarray) for state in state_series_2): # Validate series_2
            raise ValueError("Parameter 'state_series_2' must resolve to a list of numpy arrays.")


        cfp_results = action_function(state_series_1, state_series_2) # Execute CFP function
        return cfp_results # Return CFP metrics


    def _resolve_state_series_parameter(self, param_value: Any, context: Dict[str, Any]) -> List[np.ndarray]:
        """
        Resolves a state series parameter, handling dynamic references and context variables.

        If the parameter value is a string, it checks if it refers to 'get_agent_state_history'
        or a context variable. Otherwise, it assumes the value is already the state series.

        Args:
            param_value (Any): The parameter value to resolve.
            context (Dict[str, Any]): Current context variables dictionary.

        Returns:
            List[np.ndarray]: Resolved state series (list of numpy arrays).

        Raises:
            ValueError: if dynamic reference cannot be resolved or if context variable is not found.
        """
        if isinstance(param_value, str):
            if param_value == 'get_agent_state_history': # Check for agent state history reference
                return self.agent.state_history # Return agent's state history
            elif param_value in context: # Check if it's a context variable
                context_val = context[param_value] # Get context variable value
                if isinstance(context_val, ScalableAgent): # If context variable is another agent
                    return context_val.state_history # Return other agent's state history
                else:
                    raise ValueError(f"Context variable '{param_value}' is not a ScalableAgent instance for state history access.") # Error if not ScalableAgent
            else:
                raise ValueError(f"Unresolvable dynamic parameter reference: '{param_value}'.") # Error for unresolvable reference
        elif isinstance(param_value, list): # If already a list, assume it's the state series
            return param_value # Return directly
        else:
            raise ValueError("State series parameter must be a string reference or a list of state vectors.") # Error for invalid parameter type


    def execute_workflow(self, workflow: list) -> dict:
        """Execute workflow with quantum state tracking"""
        results = {}
        quantum_context = self.agent.quantum_state.copy()
        
        for step in workflow:
            try:
                # Quantum context preservation
                if 'quantum_context' in step:
                    quantum_context = self._handle_quantum_context(step, quantum_context)
                
                # Enhanced SPR processing
                if step['action'] == 'process_spr':
                    results.update(self._process_spr(step, quantum_context))
                    
                # Nanotech workflows
                elif step['action'] == 'nanomaterial_synthesis':
                    results.update(self._handle_nanotech(step))
                    
                # Quantum workflows
                elif step['action'] == 'quantum_operation':
                    results.update(self._handle_quantum_op(step, quantum_context))
                    
            except Exception as e:
                self._handle_workflow_error(step, e)
                
        return results

    def _handle_quantum_context(self, step: dict, state: np.ndarray) -> np.ndarray:
        """Process quantum context operations"""
        if step['quantum_context'] == 'entangle':
            return entangled_state(state, step.get('state_type', 'bell'))
        elif step['quantum_context'] == 'evolve':
            return quantum_state_evolution(state, step['steps'])
        return state



if __name__ == '__main__':
    # Example ScalableAgent setup (same as in scalable_framework.py example)
    operators_example = {
        'operator_A': np.array([[0.5, 0], [0, 0.5]]),
        'operator_B': np.array([[0, -1], [1, 0]]),
        'operator_C':  np.array([[1.2, 0], [0, 1.2]])
    }
    def simple_operator_selection_strategy(agent: ScalableAgent) -> str:
        operator_keys = list(agent.operators.keys())
        current_index = operator_keys.index(agent.current_operator_key)
        next_index = (current_index + 1) % len(operator_keys)
        return operator_keys[next_index]
    def explore_action(agent: ScalableAgent, operator: np.ndarray) -> np.ndarray:
        perturbation = np.random.normal(0, 0.1, size=agent.current_state.shape)
        return agent.current_state + perturbation
    def exploit_action(agent: ScalableAgent, operator: np.ndarray) -> np.ndarray:
        return potential_function(agent.current_state, operator)
    def consolidate_action(agent: ScalableAgent, operator: np.ndarray) -> np.ndarray:
        return agent.current_state * 0.9
    action_registry_example = {
        'web_search': web_search,
        'huggingface_dataset_search': huggingface_dataset_search,
        'github_project_search': github_project_search,
        'scholarly_article_search': scholarly_article_search,
        'execute_code': execute_code,
        'explore': explore_state_action,
        'exploit': exploit_potential_action,
        'consolidate': consolidate_action,
        'run_cfp': comparative_fluxual_processing # Add CFP function to action registry
    }
    workflow_modes_example = {
        'mode_1': ['explore', 'exploit'],
        'mode_2': ['consolidate', 'exploit', 'explore']
    }
    initial_state_example = np.array([1.0, 1.0])
    agent_workflow = ScalableAgent(
        agent_id='workflow_agent_001',
        initial_state=initial_state_example,
        operators=operators_example,
        action_registry=action_registry_example,
        workflow_modes=workflow_modes_example,
        operator_selection_strategy=simple_operator_selection_strategy,
        initial_operator_key='operator_A'
    )


    # Initialize WorkflowEngine
    workflow_engine = WorkflowEngine(agent_workflow, action_registry_example)


    # Example workflow JSON path (assuming cfp_workflow.json is in the same directory)
    workflow_json_path_example = 'cfp_workflow.json' # Path to workflow definition JSON


    # Create a dummy cfp_workflow.json for testing if it doesn't exist
    if not os.path.exists(workflow_json_path_example):
        dummy_workflow_json = [
          {"action": "web_search", "params": {"query": "quantum cryptography"}},
          {"action": "execute_code", "params": {"code_string": "print(2+2)"}},
          {
           "action": "run_cfp",
           "mode": "run_cfp",
           "params": {
               "state_series_1": "get_agent_state_history",
               "state_series_2": "agent_2"
            },
           "context": {"agent_2": "agent.state_history"} # Note: In real scenario, agent_2 would be another ScalableAgent instance
          }
        ]
        with open(workflow_json_path_example, 'w') as f:
            json.dump(dummy_workflow_json, f, indent=2)
        print(f"Created dummy workflow JSON file: {workflow_json_path_example}")


    # Execute workflow from JSON
    print(f"\nExecuting workflow from JSON: {workflow_json_path_example}")
    workflow_results = workflow_engine.execute_workflow_from_json(workflow_json_path_example, workflow_id='test_workflow_001')


    print("\nWorkflow Execution Results:")
    for i, result in enumerate(workflow_results):
        print(f"Step {i+1} Result: {result}") 