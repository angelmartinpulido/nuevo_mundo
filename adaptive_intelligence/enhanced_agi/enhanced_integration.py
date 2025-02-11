"""
Enhanced Integration Module for Advanced AGI Components
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .enhanced_consciousness import EnhancedConsciousnessModule, ConsciousnessState
from .enhanced_metacognition import EnhancedMetacognitionModule, MetacognitiveState
from .enhanced_reasoning import EnhancedReasoningModule, ReasoningState
from .enhanced_creativity import EnhancedCreativityModule, CreativeState
from .enhanced_memory import EnhancedMemoryModule, MemoryState
from .enhanced_optimization import EnhancedOptimizationModule, OptimizationState


@dataclass
class IntegratedState:
    consciousness: ConsciousnessState
    metacognition: MetacognitiveState
    reasoning: ReasoningState
    creativity: CreativeState
    memory: MemoryState
    optimization: OptimizationState
    integration_score: float
    synergy_level: float
    coherence_score: float


class EnhancedAGISystem(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_symbols: int = 1000,
        memory_size: int = 10000,
        num_resources: int = 8,
    ):
        super().__init__()

        # Core AGI components
        self.consciousness = EnhancedConsciousnessModule(input_dim, hidden_dim)
        self.metacognition = EnhancedMetacognitionModule(input_dim, hidden_dim)
        self.reasoning = EnhancedReasoningModule(input_dim, num_symbols, hidden_dim)
        self.creativity = EnhancedCreativityModule(input_dim, hidden_dim)
        self.memory = EnhancedMemoryModule(
            input_dim, hidden_dim, memory_size=memory_size
        )
        self.optimization = EnhancedOptimizationModule(
            input_dim, hidden_dim, num_resources
        )

        # Integration networks
        self.component_integration = nn.Sequential(
            nn.Linear(input_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.state_integration = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # 3 integration metrics
        )

        # Synergy network
        self.synergy_network = nn.Sequential(
            nn.Linear(input_dim * 7, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        symbols: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        resource_limits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, IntegratedState]:
        # Process through core components
        consciousness_out, consciousness_state = self.consciousness(x)
        metacognition_out, metacognition_state = self.metacognition(consciousness_out)
        reasoning_out, reasoning_state = self.reasoning(metacognition_out, symbols)
        creativity_out, creativity_state = self.creativity(reasoning_out)
        memory_out, memory_state = self.memory(creativity_out, context)
        optimization_out, optimization_state = self.optimization(
            memory_out, resource_limits
        )

        # Component integration
        integration_input = torch.cat(
            [
                consciousness_out,
                metacognition_out,
                reasoning_out,
                creativity_out,
                memory_out,
                optimization_out,
            ],
            dim=-1,
        )
        integrated_output = self.component_integration(integration_input)

        # State integration
        state_input = torch.cat(
            [
                consciousness_out,
                metacognition_out,
                reasoning_out,
                creativity_out,
                memory_out,
                optimization_out,
            ],
            dim=-1,
        )
        integration_metrics = self.state_integration(state_input)

        # Compute synergy
        synergy_input = torch.cat(
            [
                integrated_output,
                consciousness_out,
                metacognition_out,
                reasoning_out,
                creativity_out,
                memory_out,
                optimization_out,
            ],
            dim=-1,
        )
        synergy_score = self.synergy_network(synergy_input)

        # Create integrated state
        integrated_state = IntegratedState(
            consciousness=consciousness_state,
            metacognition=metacognition_state,
            reasoning=reasoning_state,
            creativity=creativity_state,
            memory=memory_state,
            optimization=optimization_state,
            integration_score=float(integration_metrics[0].sigmoid()),
            synergy_level=float(synergy_score.mean()),
            coherence_score=float(integration_metrics[1:].mean()),
        )

        return integrated_output, integrated_state

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return comprehensive metrics from all components"""
        return {
            "consciousness": self.consciousness.get_metrics(),
            "metacognition": self.metacognition.get_metrics(),
            "reasoning": self.reasoning.get_metrics(),
            "creativity": self.creativity.get_metrics(),
            "memory": self.memory.get_metrics(),
            "optimization": self.optimization.get_metrics(),
        }

    def reset_metrics(self):
        """Reset all component metrics"""
        self.memory.clear_statistics()
        self.optimization.reset_metrics()

    def get_component_states(self) -> Dict[str, Any]:
        """Return current states of all components"""
        return {
            "consciousness": self.consciousness.state_dict(),
            "metacognition": self.metacognition.state_dict(),
            "reasoning": self.reasoning.state_dict(),
            "creativity": self.creativity.state_dict(),
            "memory": self.memory.state_dict(),
            "optimization": self.optimization.state_dict(),
        }

    def load_component_states(self, states: Dict[str, Any]):
        """Load states into all components"""
        self.consciousness.load_state_dict(states["consciousness"])
        self.metacognition.load_state_dict(states["metacognition"])
        self.reasoning.load_state_dict(states["reasoning"])
        self.creativity.load_state_dict(states["creativity"])
        self.memory.load_state_dict(states["memory"])
        self.optimization.load_state_dict(states["optimization"])
