"""
Enhanced Metacognition Module with Advanced Learning and Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MetacognitiveState:
    confidence: float
    uncertainty: float
    learning_progress: float
    decision_quality: float
    introspection_depth: float
    adaptation_rate: float


class ReinforcementCore(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value estimation
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)
        state_value = self.critic(state)
        value_estimate = self.value(state)
        return action_probs, state_value, value_estimate


class TemporalProcessing(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
        )

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Tuple]:
        # Temporal convolution
        conv_out = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)

        # LSTM processing
        lstm_out, hidden = self.lstm(conv_out, hidden)

        # Temporal attention
        attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)

        return attended, hidden


class DecisionMaking(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()

        # Decision network
        self.decision_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Uncertainty estimation
        self.uncertainty = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Sigmoid(),
        )

        # Value estimation
        self.value_estimate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decisions = self.decision_network(x)
        uncertainties = self.uncertainty(x)
        values = self.value_estimate(x)
        return F.softmax(decisions, dim=-1), uncertainties, values


class Introspection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Self-analysis network
        self.self_analysis = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Error detection
        self.error_detector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        analysis = self.self_analysis(x)
        combined = torch.cat([x, analysis], dim=-1)
        confidence = self.confidence(combined)
        error_prob = self.error_detector(combined)
        return analysis, confidence, error_prob


class UncertaintyAnalysis(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Aleatoric, epistemic, and total uncertainty
        )

        # Risk assessment
        self.risk_assessor = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uncertainties = self.uncertainty_estimator(x)
        risk = self.risk_assessor(torch.cat([x, uncertainties], dim=-1))
        return uncertainties, risk


class EnhancedMetacognitionModule(nn.Module):
    def __init__(
        self, input_dim: int = 512, num_actions: int = 10, hidden_dim: int = 256
    ):
        super().__init__()

        # Core components
        self.reinforcement_core = ReinforcementCore(input_dim, num_actions, hidden_dim)
        self.temporal_processing = TemporalProcessing(input_dim, hidden_dim)
        self.decision_making = DecisionMaking(input_dim * 2, num_actions, hidden_dim)
        self.introspection = Introspection(input_dim, hidden_dim)
        self.uncertainty_analysis = UncertaintyAnalysis(input_dim, hidden_dim)

        # Integration network
        self.integration = nn.Sequential(
            nn.Linear(input_dim * 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # State assessment
        self.state_assessment = nn.Sequential(
            nn.Linear(input_dim * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # 6 metacognitive state metrics
        )

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, MetacognitiveState]:
        # Reinforcement learning
        action_probs, state_value, value_est = self.reinforcement_core(x)

        # Temporal processing
        temporal_features, new_hidden = self.temporal_processing(x, hidden_state)

        # Decision making
        decisions, decision_uncertainty, decision_values = self.decision_making(
            torch.cat([x, temporal_features], dim=-1)
        )

        # Introspection
        self_analysis, confidence, error_prob = self.introspection(x)

        # Uncertainty analysis
        uncertainties, risk = self.uncertainty_analysis(x)

        # Integration
        integration_input = torch.cat(
            [
                x,
                temporal_features,
                self_analysis,
                action_probs,
                decisions,
                uncertainties,
            ],
            dim=-1,
        )
        integrated = self.integration(integration_input)

        # State assessment
        state_input = torch.cat(
            [
                x,
                temporal_features,
                self_analysis,
                action_probs,
                decisions,
                uncertainties,
                integrated,
            ],
            dim=-1,
        )
        state_metrics = self.state_assessment(state_input)

        # Create metacognitive state
        metacognitive_state = MetacognitiveState(
            confidence=float(confidence.mean()),
            uncertainty=float(uncertainties.mean()),
            learning_progress=float(value_est.mean()),
            decision_quality=float(decision_values.mean()),
            introspection_depth=float(self_analysis.std()),
            adaptation_rate=float(action_probs.std()),
        )

        return integrated, metacognitive_state

    def get_metrics(self) -> Dict[str, float]:
        """Return current metacognition metrics"""
        return {
            "confidence": float(
                torch.mean(self.introspection.confidence(torch.randn(1, 512))).item()
            ),
            "uncertainty": float(
                torch.mean(
                    self.uncertainty_analysis.uncertainty_estimator(torch.randn(1, 512))
                ).item()
            ),
            "decision_quality": float(
                torch.mean(
                    self.decision_making.value_estimate(torch.randn(1, 512))
                ).item()
            ),
            "learning_efficiency": float(
                torch.mean(self.reinforcement_core.value(torch.randn(1, 512))).item()
            ),
        }
