"""
Enhanced Creativity Module with Advanced Generation and Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from enum import Enum


class CreativeMode(Enum):
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"
    ASSOCIATIVE = "associative"
    TRANSFORMATIVE = "transformative"
    EXPLORATORY = "exploratory"


@dataclass
class CreativeState:
    mode: CreativeMode
    novelty_score: float
    originality: float
    usefulness: float
    coherence: float
    diversity: float
    exploration_depth: float


class DivergentThinking(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_branches: int = 8):
        super().__init__()

        # Enhanced parallel creative pathways with residual connections
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(num_branches)
            ]
        )

        # Multi-head attention with improved feature extraction
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)

        # Enhanced branch selection with dynamic weighting
        self.branch_selector = nn.Sequential(
            nn.Linear(input_dim * num_branches, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_branches),
            nn.Softmax(dim=-1),
        )

        # Advanced integration with hierarchical processing
        self.integration = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim * num_branches, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(3)  # Multiple integration levels
            ]
        )

        # Meta-integration network
        self.meta_integration = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Weights for different integration levels
            nn.Softmax(dim=-1),
        )

        # Novelty enhancer
        self.novelty_enhancer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Generate multiple creative pathways
        branch_outputs = [branch(x) for branch in self.branches]
        branch_tensor = torch.stack(branch_outputs, dim=1)

        # Apply attention mechanism
        attended, attention_weights = self.attention(
            branch_tensor, branch_tensor, branch_tensor
        )

        # Dynamic branch selection
        selection_input = branch_tensor.view(branch_tensor.size(0), -1)
        branch_weights = self.branch_selector(selection_input)

        # Multi-level integration
        integration_outputs = []
        for integrator in self.integration:
            integration_input = attended.view(attended.size(0), -1)
            integration_outputs.append(integrator(integration_input))

        # Meta-integration
        meta_input = torch.cat(integration_outputs, dim=-1)
        integration_weights = self.meta_integration(meta_input)
        integrated = sum(
            w * o
            for w, o in zip(integration_weights.unbind(dim=-1), integration_outputs)
        )

        # Enhance novelty
        novelty_input = torch.cat([integrated, x], dim=-1)
        enhanced = self.novelty_enhancer(novelty_input)

        # Return output and metadata
        metadata = {
            "branch_outputs": branch_tensor,
            "attention_weights": attention_weights,
            "branch_weights": branch_weights,
            "integration_weights": integration_weights,
        }

        return enhanced, metadata


class AssociativeNetwork(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 256, num_associations: int = 5
    ):
        super().__init__()

        # Enhanced association embeddings with learned priors
        self.association_embedding = nn.Parameter(
            torch.randn(num_associations, hidden_dim)
        )
        self.embedding_prior = nn.Parameter(torch.randn(1, hidden_dim))

        # Advanced association generator with hierarchical processing
        self.generator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(3)  # Multiple generation levels
            ]
        )

        # Enhanced combination network with attention
        self.combiner = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(3)  # Multiple combination levels
            ]
        )

        # Association attention mechanism
        self.association_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # Meta-generation network
        self.meta_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Weights for different generation levels
            nn.Softmax(dim=-1),
        )

        # Association strength modulator
        self.strength_modulator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Novelty filter
        self.novelty_filter = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = x.size(0)

        # Generate embeddings with prior influence
        prior_expanded = self.embedding_prior.expand(
            self.association_embedding.size(0), -1
        )
        modulated_embedding = self.association_embedding * torch.sigmoid(prior_expanded)
        embeddings = modulated_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-level association generation
        generation_outputs = []
        for generator in self.generator:
            x_expanded = x.unsqueeze(1).expand(-1, embeddings.size(1), -1)
            combined = torch.cat([x_expanded, embeddings], dim=-1)
            generation_outputs.append(generator(combined))

        # Apply attention to associations
        attended_associations, attention_weights = self.association_attention(
            torch.stack(generation_outputs, dim=0),
            torch.stack(generation_outputs, dim=0),
            torch.stack(generation_outputs, dim=0),
        )

        # Meta-generation
        meta_input = torch.cat([g.mean(1) for g in generation_outputs], dim=-1)
        generation_weights = self.meta_generator(meta_input)

        # Combine associations with attention
        final_associations = []
        for i, combiner in enumerate(self.combiner):
            combined = torch.cat(
                [
                    attended_associations[i],
                    x.unsqueeze(1).expand(-1, attended_associations.size(1), -1),
                ],
                dim=-1,
            )
            final_associations.append(combiner(combined))

        # Modulate association strength
        strength = self.strength_modulator(
            torch.cat([attended_associations.mean((0, 1)), x], dim=-1)
        )

        # Apply novelty filter
        novelty_input = torch.cat(
            [
                sum(
                    w * f.mean(1)
                    for w, f in zip(
                        generation_weights.unbind(dim=-1), final_associations
                    )
                ),
                x,
            ],
            dim=-1,
        )
        filtered = self.novelty_filter(novelty_input)

        # Return output and metadata
        metadata = {
            "attention_weights": attention_weights,
            "generation_weights": generation_weights,
            "association_strength": strength,
            "embeddings": modulated_embedding,
        }

        return filtered, metadata


class NoveltyEvaluation(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Enhanced novelty detector with multi-scale analysis
        self.novelty_network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(3)  # Multiple novelty scales
            ]
        )

        # Advanced quality evaluator with hierarchical assessment
        self.quality_network = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(3)  # Multiple quality aspects
            ]
        )

        # Enhanced coherence checker with attention
        self.coherence_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Coherence attention
        self.coherence_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # Meta-evaluation network
        self.meta_evaluation = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Weights for different evaluation aspects
            nn.Softmax(dim=-1),
        )

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Adaptive threshold
        self.threshold = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Thresholds for novelty, quality, coherence
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Multi-scale novelty detection
        novelty_scores = torch.stack([net(x) for net in self.novelty_network], dim=1)

        # Multi-aspect quality evaluation
        quality_scores = torch.stack([net(x) for net in self.quality_network], dim=1)

        # Enhanced coherence checking
        coherence_features = self.coherence_network(x)
        coherence_attended, attention_weights = self.coherence_attention(
            coherence_features.unsqueeze(0),
            coherence_features.unsqueeze(0),
            coherence_features.unsqueeze(0),
        )

        # Meta-evaluation
        meta_input = torch.cat(
            [
                novelty_scores.mean(1),
                quality_scores.mean(1),
                coherence_attended.squeeze(0),
            ],
            dim=-1,
        )
        evaluation_weights = self.meta_evaluation(meta_input)

        # Compute confidence
        confidence_input = torch.cat(
            [
                novelty_scores.mean(1),
                quality_scores.mean(1),
                coherence_attended.squeeze(0),
                evaluation_weights,
            ],
            dim=-1,
        )
        confidence_score = self.confidence(confidence_input)

        # Compute adaptive thresholds
        thresholds = self.threshold(x)

        # Apply thresholds and weights
        final_novelty = (novelty_scores > thresholds[:, 0:1]).float() * novelty_scores
        final_quality = (quality_scores > thresholds[:, 1:2]).float() * quality_scores
        final_coherence = (
            coherence_attended.squeeze(0) > thresholds[:, 2:3]
        ).float() * coherence_attended.squeeze(0)

        # Return scores and metadata
        metadata = {
            "novelty_scores": novelty_scores,
            "quality_scores": quality_scores,
            "coherence_attention": attention_weights,
            "evaluation_weights": evaluation_weights,
            "confidence": confidence_score,
            "thresholds": thresholds,
        }

        return final_novelty.mean(1), final_quality.mean(1), final_coherence, metadata


class ConceptCombination(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Enhanced concept encoder with hierarchical processing
        self.concept_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(3)  # Multiple encoding levels
            ]
        )

        # Advanced combination network with multi-head attention
        self.combination_network = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                )
                for _ in range(4)
            ]
        )

        # Enhanced blend generator with residual connections
        self.blend_generator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(3)  # Multiple generation levels
            ]
        )

        # Concept attention mechanism
        self.concept_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # Meta-combination network
        self.meta_combination = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Weights for different combination levels
            nn.Softmax(dim=-1),
        )

        # Blend coherence checker
        self.coherence_checker = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Adaptive mixing controller
        self.mixing_controller = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, concepts: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Multi-level concept encoding
        encoded_concepts = []
        for encoder in self.concept_encoder:
            encoded = encoder(concepts)
            encoded_concepts.append(encoded)

        # Apply attention to encoded concepts
        attended_concepts = []
        attention_weights = []
        for encoded in encoded_concepts:
            attended, weights = self.concept_attention(
                encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0)
            )
            attended_concepts.append(attended.squeeze(0))
            attention_weights.append(weights)

        # Multi-layer combination processing
        combined_concepts = []
        for layer in self.combination_network:
            for i in range(len(attended_concepts)):
                attended_concepts[i] = layer(attended_concepts[i].unsqueeze(0)).squeeze(
                    0
                )
            combined_concepts.append(torch.stack(attended_concepts, dim=0).mean(0))

        # Generate blends at multiple levels
        blends = []
        for generator, combined in zip(self.blend_generator, combined_concepts):
            blend = generator(combined)
            blends.append(blend)

        # Meta-combination
        meta_input = torch.cat([b.mean(1) for b in torch.stack(blends)], dim=-1)
        combination_weights = self.meta_combination(meta_input)

        # Check coherence
        coherence_scores = []
        for blend in blends:
            coherence_input = torch.cat([blend, concepts], dim=-1)
            coherence_scores.append(self.coherence_checker(coherence_input))

        # Adaptive mixing
        mixing_weights = []
        for blend in blends:
            mix_input = torch.cat([blend.mean(1), concepts.mean(1)], dim=-1)
            mixing_weights.append(self.mixing_controller(mix_input))

        # Final blend integration
        weighted_blends = [
            w * b for w, b in zip(combination_weights.unbind(dim=-1), blends)
        ]
        coherence_weighted = [c * b for c, b in zip(coherence_scores, weighted_blends)]
        mixing_adjusted = [m * b for m, b in zip(mixing_weights, coherence_weighted)]

        final_blend = sum(mixing_adjusted) / len(mixing_adjusted)

        # Return blends and metadata
        metadata = {
            "attention_weights": attention_weights,
            "combination_weights": combination_weights,
            "coherence_scores": torch.stack(coherence_scores),
            "mixing_weights": torch.stack(mixing_weights),
            "intermediate_blends": blends,
        }

        return final_blend, metadata


class ExplorationModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()

        # Enhanced encoder with hierarchical processing
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, latent_dim * 2),
                )
                for _ in range(3)  # Multiple encoding levels
            ]
        )

        # Advanced decoder with residual connections
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, input_dim),
                    nn.LayerNorm(input_dim),
                )
                for _ in range(3)  # Multiple decoding levels
            ]
        )

        # Latent space attention
        self.latent_attention = nn.MultiheadAttention(latent_dim, num_heads=4)

        # Meta-exploration network
        self.meta_exploration = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),  # Weights for different exploration levels
            nn.Softmax(dim=-1),
        )

        # Exploration controller
        self.exploration_controller = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Novelty discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Multi-level encoding
        encoded_params = []
        for encoder in self.encoder:
            h = encoder(x)
            mu, logvar = h.chunk(2, dim=-1)
            z = self.reparameterize(mu, logvar)
            encoded_params.append((z, mu, logvar))

        # Apply attention to latent representations
        latent_stack = torch.stack([p[0] for p in encoded_params], dim=0)
        attended_latent, attention_weights = self.latent_attention(
            latent_stack, latent_stack, latent_stack
        )

        # Multi-level decoding
        decoded = []
        for decoder, (z, _, _) in zip(self.decoder, encoded_params):
            decoded.append(decoder(z))

        # Meta-exploration
        meta_input = torch.cat([p[0] for p in encoded_params], dim=-1)
        exploration_weights = self.meta_exploration(meta_input)

        # Control exploration degree
        exploration_input = torch.cat(
            [
                attended_latent.mean(0),
                torch.stack([p[0] for p in encoded_params]).mean(0),
            ],
            dim=-1,
        )
        exploration_factor = self.exploration_controller(exploration_input)

        # Apply exploration and combine results
        explored = []
        for dec, weight in zip(decoded, exploration_weights.unbind(dim=-1)):
            noise = torch.randn_like(dec) * exploration_factor
            explored.append(dec + noise * weight)

        # Discriminate novelty
        novelty_scores = [self.discriminator(e) for e in explored]

        # Combine results based on novelty and exploration weights
        final_output = sum(s * e for s, e in zip(novelty_scores, explored)) / len(
            explored
        )

        # Return output and metadata
        metadata = {
            "latent_params": [(mu, logvar) for _, mu, logvar in encoded_params],
            "attention_weights": attention_weights,
            "exploration_weights": exploration_weights,
            "exploration_factor": exploration_factor,
            "novelty_scores": torch.stack(novelty_scores),
            "decoded_states": decoded,
        }

        return final_output, metadata


class EnhancedCreativityModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_branches: int = 8,
        num_associations: int = 5,
        latent_dim: int = 64,
    ):
        super().__init__()

        # Core creative components
        self.divergent_thinking = DivergentThinking(input_dim, hidden_dim, num_branches)
        self.associative_network = AssociativeNetwork(
            input_dim, hidden_dim, num_associations
        )
        self.novelty_evaluation = NoveltyEvaluation(input_dim, hidden_dim)
        self.concept_combination = ConceptCombination(input_dim, hidden_dim)
        self.exploration = ExplorationModule(input_dim, hidden_dim, latent_dim)

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
            nn.Linear(hidden_dim, 7),  # 7 creativity state metrics
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, CreativeState]:
        # Divergent thinking
        divergent_out, branch_outputs = self.divergent_thinking(x)

        # Associative thinking
        associations, assoc_features = self.associative_network(x)

        # Concept combination
        concepts, combined = self.concept_combination(x.unsqueeze(1))

        # Exploration
        explored, mu, logvar = self.exploration(x)

        # Novelty evaluation
        novelty, quality, coherence = self.novelty_evaluation(
            torch.cat(
                [divergent_out, associations.mean(1), concepts.squeeze(1), explored],
                dim=-1,
            )
        )

        # Integration
        integration_input = torch.cat(
            [x, divergent_out, associations.mean(1), concepts.squeeze(1), explored, mu],
            dim=-1,
        )
        integrated = self.integration(integration_input)

        # State assessment
        state_input = torch.cat(
            [
                x,
                divergent_out,
                associations.mean(1),
                concepts.squeeze(1),
                explored,
                mu,
                integrated,
            ],
            dim=-1,
        )
        state_metrics = self.state_assessment(state_input)

        # Create creative state
        creative_state = CreativeState(
            mode=CreativeMode(torch.argmax(state_metrics[0:5]).item()),
            novelty_score=float(novelty.mean()),
            originality=float(state_metrics[5].sigmoid()),
            usefulness=float(quality.mean()),
            coherence=float(coherence.mean()),
            diversity=float(torch.std(torch.stack(branch_outputs, dim=1))),
            exploration_depth=float(-logvar.mean()),
        )

        return integrated, creative_state

    def get_metrics(self) -> Dict[str, float]:
        """Return current creativity metrics"""
        return {
            "divergent_strength": float(
                torch.mean(
                    self.divergent_thinking.integration(torch.randn(1, 2048))
                ).item()
            ),
            "associative_richness": float(
                torch.std(
                    self.associative_network.generator(torch.randn(1, 768))
                ).item()
            ),
            "novelty_level": float(
                torch.mean(
                    self.novelty_evaluation.novelty_network(torch.randn(1, 512))
                ).item()
            ),
            "concept_blend_quality": float(
                torch.mean(
                    self.concept_combination.blend_generator(torch.randn(1, 256))
                ).item()
            ),
            "exploration_diversity": float(
                torch.std(self.exploration.decoder(torch.randn(1, 64))).item()
            ),
        }
