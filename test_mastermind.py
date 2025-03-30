#!/usr/bin/env python3
"""
Test script for MasterMind_AI system
Verifies core functionality and integration of components
"""

import unittest
import numpy as np
from code.mastermind_ai import MasterMind_AI
from code.system_orchestrator import SystemOrchestrator
from code.workflow_engine import WorkflowEngine
from code.cfp_framework import comparative_flux_density
from code.quantum_utils import superposition_state, entangled_state

class TestMasterMindAI(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        initial_state = np.array([0.5, 0.5, 0.5, 0.5])  # 4-dimensional state
        self.mastermind = MasterMind_AI(
            agent_id='test_agent',
            initial_state=initial_state,
            resonance_threshold=0.75
        )
        self.mastermind.quantum_state = initial_state  # Explicitly set quantum state
        self.mastermind.state_history = [initial_state]  # Initialize state history
        self.orchestrator = SystemOrchestrator(mastermind_ai=self.mastermind)
        
    def test_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.mastermind)
        self.assertIsNotNone(self.mastermind.kg)
        self.assertIsNotNone(self.mastermind.quantum_state)
        self.assertEqual(len(self.mastermind.sprs), 5)  # Check number of resonant states
        
    def test_knowledge_graph(self):
        """Test knowledge graph operations"""
        # Add test concept
        self.mastermind.add_knowledge(
            concept="TestConcept",
            concept_type="test",
            connections=["ResonantiA"]
        )
        
        # Verify concept was added
        self.assertTrue("TestConcept" in self.mastermind.kg.nodes())
        self.assertTrue(self.mastermind.kg.has_edge("TestConcept", "ResonantiA"))
        
    def test_query_processing(self):
        """Test query processing"""
        result = self.mastermind.process_query("What is quantum computing?")
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('response', result)
        self.assertIn('resonance_score', result)
        self.assertIn('thinking', result)
        
    def test_quantum_operations(self):
        """Test quantum computing operations"""
        # Test superposition state
        state = superposition_state(4, [0, 2])  # Create superposition of |0⟩ and |2⟩ states
        self.assertEqual(len(state), 4)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0)
        
        # Test entangled state
        ent_state = entangled_state(2)
        self.assertEqual(len(ent_state), 4)
        self.assertAlmostEqual(np.sum(np.abs(ent_state)**2), 1.0)
        
    def test_cfp_operations(self):
        """Test Comparative Fluxual Processing operations"""
        # Create test states
        state1 = np.array([0.707, 0.707])
        state2 = np.array([0.707, -0.707])
        
        # Calculate flux density
        flux = comparative_flux_density(state1, state2)
        self.assertIsInstance(flux, float)
        self.assertGreaterEqual(flux, 0.0)
        
    def test_workflow_execution(self):
        """Test workflow execution"""
        # Create test workflow
        workflow = [
            {
                "action": "web_search",
                "params": {"query": "quantum computing"}
            }
        ]
        
        # Initialize workflow engine with quantum state
        engine = WorkflowEngine(self.mastermind, {})
        self.mastermind.quantum_state = np.array([0.5, 0.5, 0.5, 0.5])  # Initialize quantum state
        
        # Execute workflow
        results = engine.execute_workflow(workflow)
        self.assertIsInstance(results, dict)  # Changed from list to dict to match implementation
        
    def test_system_orchestration(self):
        """Test system orchestration"""
        # Initialize quantum system with a 4-dimensional state
        self.mastermind.quantum_state = np.array([0.5, 0.5, 0.5, 0.5])
        self.orchestrator.initialize_quantum_system()
        
        # Execute cognitive workflow
        result = self.orchestrator.execute_cognitive_workflow(
            "How can quantum computing improve AI?"
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('quantum_metrics', result)
        self.assertIn('concept_map', result)
        
    def test_tool_registration(self):
        """Test tool registration and usage"""
        # Create test tool
        def test_tool(**params):
            return {"result": "Test tool executed"}
            
        # Register tool
        self.mastermind.register_tool(test_tool)
        
        # Verify tool registration
        self.assertIn('test_tool', self.mastermind.tools)
        
        # Test tool execution
        result = self.mastermind.tools['test_tool']()
        self.assertEqual(result['result'], "Test tool executed")

if __name__ == '__main__':
    unittest.main() 