"""
Fragment optimizer for P2P system.
Handles fragment optimization, size reduction, and efficiency improvements.
"""

import asyncio
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import lz4.frame
import zstandard as zstd
import xxhash
from concurrent.futures import ThreadPoolExecutor
import ast
import autopep8
import black
from pyminifier import minification
import networkx as nx


class OptimizationType(Enum):
    MINIMAL = 0  # Basic optimizations
    BALANCED = 1  # Balance between size and functionality
    AGGRESSIVE = 2  # Maximum size reduction


@dataclass
class OptimizationMetrics:
    original_size: int
    optimized_size: int
    compression_ratio: float
    functionality_score: float
    performance_impact: float
    memory_usage: float


class FragmentOptimizer:
    def __init__(self):
        self.optimization_cache: Dict[str, bytes] = {}
        self.metrics_history: Dict[str, List[OptimizationMetrics]] = {}
        self.dependency_graph = nx.DiGraph()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Optimization thresholds
        self.MAX_FRAGMENT_SIZE = 100 * 1024  # 100KB
        self.TARGET_COMPRESSION_RATIO = 0.5
        self.MIN_FUNCTIONALITY_SCORE = 0.9

    async def optimize_fragment(
        self, fragment_id: str, code: bytes, opt_type: OptimizationType
    ) -> bytes:
        """Optimize fragment code for maximum efficiency."""
        try:
            # Check cache first
            cache_key = f"{fragment_id}:{opt_type.value}"
            if cache_key in self.optimization_cache:
                return self.optimization_cache[cache_key]

            # Decode code
            code_str = code.decode("utf-8")

            # Parse and analyze code
            tree = ast.parse(code_str)

            # Apply optimizations based on type
            if opt_type == OptimizationType.MINIMAL:
                optimized = await self._apply_minimal_optimizations(tree, code_str)
            elif opt_type == OptimizationType.BALANCED:
                optimized = await self._apply_balanced_optimizations(tree, code_str)
            else:
                optimized = await self._apply_aggressive_optimizations(tree, code_str)

            # Compress optimized code
            compressed = await self._compress_code(optimized.encode())

            # Cache result
            self.optimization_cache[cache_key] = compressed

            # Collect metrics
            metrics = OptimizationMetrics(
                original_size=len(code),
                optimized_size=len(compressed),
                compression_ratio=len(compressed) / len(code),
                functionality_score=await self._measure_functionality(optimized),
                performance_impact=await self._measure_performance_impact(optimized),
                memory_usage=await self._measure_memory_usage(optimized),
            )

            self.metrics_history.setdefault(fragment_id, []).append(metrics)

            return compressed

        except Exception as e:
            logging.error(f"Fragment optimization failed: {e}")
            return code

    async def _apply_minimal_optimizations(self, tree: ast.AST, code: str) -> str:
        """Apply basic optimizations that preserve all functionality."""
        optimizations = [
            self._remove_comments,
            self._optimize_imports,
            self._format_code,
            self._optimize_whitespace,
        ]

        return await self._apply_optimizations(code, optimizations)

    async def _apply_balanced_optimizations(self, tree: ast.AST, code: str) -> str:
        """Apply moderate optimizations balancing size and functionality."""
        optimizations = [
            self._remove_comments,
            self._optimize_imports,
            self._format_code,
            self._optimize_whitespace,
            self._optimize_variables,
            self._merge_functions,
            self._optimize_loops,
            self._remove_unused_code,
        ]

        return await self._apply_optimizations(code, optimizations)

    async def _apply_aggressive_optimizations(self, tree: ast.AST, code: str) -> str:
        """Apply maximum optimizations focusing on size reduction."""
        optimizations = [
            self._remove_comments,
            self._optimize_imports,
            self._format_code,
            self._optimize_whitespace,
            self._optimize_variables,
            self._merge_functions,
            self._optimize_loops,
            self._remove_unused_code,
            self._minimize_code,
            self._optimize_constants,
            self._inline_functions,
            self._reduce_abstractions,
        ]

        return await self._apply_optimizations(code, optimizations)

    async def _apply_optimizations(self, code: str, optimizations: List[Any]) -> str:
        """Apply a series of optimizations to the code."""
        optimized = code

        for opt in optimizations:
            try:
                optimized = await opt(optimized)
            except Exception as e:
                logging.error(f"Optimization {opt.__name__} failed: {e}")

        return optimized

    async def _remove_comments(self, code: str) -> str:
        """Remove comments and docstrings."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                # Remove docstrings
                node.value.s = ""
        return ast.unparse(tree)

    async def _optimize_imports(self, code: str) -> str:
        """Optimize and minimize imports."""
        tree = ast.parse(code)
        imports = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.name] = name.asname
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for name in node.names:
                    imports[f"{module}.{name.name}"] = name.asname

        # Rebuild optimized imports
        import_lines = []
        for module, alias in imports.items():
            if alias:
                import_lines.append(f"import {module} as {alias}")
            else:
                import_lines.append(f"import {module}")

        return "\n".join(import_lines) + "\n" + code

    async def _format_code(self, code: str) -> str:
        """Format code for minimal size while maintaining readability."""
        try:
            # Use black for consistent formatting
            formatted = black.format_str(code, mode=black.FileMode())
            # Use autopep8 for additional optimizations
            formatted = autopep8.fix_code(formatted, options={"aggressive": 1})
            return formatted
        except Exception:
            return code

    async def _optimize_whitespace(self, code: str) -> str:
        """Optimize whitespace while maintaining minimal readability."""
        lines = code.split("\n")
        optimized_lines = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            # Normalize indentation to single spaces
            line = line.lstrip().replace("    ", " ")
            if line:
                optimized_lines.append(line)

        return "\n".join(optimized_lines)

    async def _optimize_variables(self, code: str) -> str:
        """Optimize variable names and usage."""
        tree = ast.parse(code)
        variable_map = {}
        counter = 0

        class VariableOptimizer(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in variable_map:
                        nonlocal counter
                        variable_map[node.id] = f"v{counter}"
                        counter += 1
                return ast.Name(id=variable_map.get(node.id, node.id), ctx=node.ctx)

        optimized_tree = VariableOptimizer().visit(tree)
        return ast.unparse(optimized_tree)

    async def _merge_functions(self, code: str) -> str:
        """Merge related functions when possible."""
        tree = ast.parse(code)
        function_groups = {}

        # Group related functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Analyze function similarity and group
                signature = self._get_function_signature(node)
                function_groups.setdefault(signature, []).append(node)

        # Merge similar functions
        for group in function_groups.values():
            if len(group) > 1:
                await self._merge_similar_functions(group)

        return ast.unparse(tree)

    async def _optimize_loops(self, code: str) -> str:
        """Optimize loop structures."""
        tree = ast.parse(code)

        class LoopOptimizer(ast.NodeTransformer):
            def visit_For(self, node):
                # Convert to more efficient form if possible
                if (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):
                    # Optimize range() calls
                    return self._optimize_range_loop(node)
                return node

            def _optimize_range_loop(self, node):
                # Convert to while loop if more efficient
                if len(node.iter.args) == 1:
                    return ast.While(
                        test=ast.Compare(
                            left=node.target,
                            ops=[ast.Lt()],
                            comparators=[node.iter.args[0]],
                        ),
                        body=node.body
                        + [
                            ast.AugAssign(
                                target=node.target, op=ast.Add(), value=ast.Num(n=1)
                            )
                        ],
                        orelse=[],
                    )
                return node

        return ast.unparse(LoopOptimizer().visit(tree))

    async def _remove_unused_code(self, code: str) -> str:
        """Remove unused code while preserving functionality."""
        tree = ast.parse(code)
        used_names = set()

        # Collect used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)

        # Remove unused definitions
        class UnusedRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name not in used_names:
                    return None
                return node

            def visit_ClassDef(self, node):
                if node.name not in used_names:
                    return None
                return node

            def visit_Assign(self, node):
                if (
                    isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id not in used_names
                ):
                    return None
                return node

        return ast.unparse(UnusedRemover().visit(tree))

    async def _minimize_code(self, code: str) -> str:
        """Apply aggressive code minimization."""
        try:
            return minification.minify(code)
        except:
            return code

    async def _optimize_constants(self, code: str) -> str:
        """Optimize constant values and expressions."""
        tree = ast.parse(code)

        class ConstantOptimizer(ast.NodeTransformer):
            def visit_BinOp(self, node):
                # Evaluate constant expressions
                if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
                    try:
                        result = eval(
                            f"{node.left.n} {type(node.op).__name__} {node.right.n}"
                        )
                        return ast.Num(n=result)
                    except:
                        pass
                return node

        return ast.unparse(ConstantOptimizer().visit(tree))

    async def _inline_functions(self, code: str) -> str:
        """Inline small functions for better performance."""
        tree = ast.parse(code)

        class FunctionInliner(ast.NodeTransformer):
            def __init__(self):
                self.functions = {}

            def visit_FunctionDef(self, node):
                # Store function definition
                self.functions[node.name] = node
                # Only keep non-inlinable functions
                if len(node.body) > 3:  # Don't inline large functions
                    return node
                return None

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                    func = self.functions[node.func.id]
                    if len(func.body) <= 3:  # Inline small functions
                        return self._inline_function(func, node.args)
                return node

            def _inline_function(self, func, args):
                # Create parameter mapping
                param_map = dict(zip((a.arg for a in func.args.args), args))

                # Replace parameters with arguments
                class ParameterReplacer(ast.NodeTransformer):
                    def visit_Name(self, node):
                        if node.id in param_map:
                            return param_map[node.id]
                        return node

                # Inline function body
                new_body = []
                for stmt in func.body[:-1]:
                    new_body.append(ParameterReplacer().visit(stmt))
                return new_body + [ParameterReplacer().visit(func.body[-1].value)]

        return ast.unparse(FunctionInliner().visit(tree))

    async def _reduce_abstractions(self, code: str) -> str:
        """Reduce abstraction layers while maintaining functionality."""
        tree = ast.parse(code)

        class AbstractionReducer(ast.NodeTransformer):
            def visit_ClassDef(self, node):
                # Flatten simple class hierarchies
                if len(node.bases) == 1 and isinstance(node.bases[0], ast.Name):
                    return self._flatten_class(node)
                return node

            def _flatten_class(self, node):
                # Combine parent and child class
                parent_name = node.bases[0].id
                if parent_name in self.parent_classes:
                    parent = self.parent_classes[parent_name]
                    # Merge methods and attributes
                    node.body = parent.body + node.body
                    node.bases = parent.bases
                return node

        return ast.unparse(AbstractionReducer().visit(tree))

    async def _compress_code(self, code: bytes) -> bytes:
        """Compress code using optimal compression."""
        # Try different compression methods
        compressed_lz4 = lz4.frame.compress(code)

        cctx = zstd.ZstdCompressor(level=22)
        compressed_zstd = cctx.compress(code)

        # Use the smallest result
        if len(compressed_lz4) < len(compressed_zstd):
            return compressed_lz4
        return compressed_zstd

    async def _measure_functionality(self, code: str) -> float:
        """Measure code functionality preservation."""
        try:
            # Parse and analyze code
            tree = ast.parse(code)

            # Check for critical structures
            functionality_checks = [
                self._check_syntax_validity(tree),
                self._check_semantic_validity(tree),
                self._check_api_compatibility(tree),
                self._check_error_handling(tree),
            ]

            return sum(functionality_checks) / len(functionality_checks)

        except Exception:
            return 0.0

    async def _measure_performance_impact(self, code: str) -> float:
        """Measure performance impact of optimizations."""
        try:
            # Compile code
            compiled = compile(code, "<string>", "exec")

            # Measure execution time
            start_time = time.perf_counter()
            exec(compiled)
            end_time = time.perf_counter()

            # Calculate impact (lower is better)
            return min(1.0, (end_time - start_time) / 0.1)

        except Exception:
            return 1.0

    async def _measure_memory_usage(self, code: str) -> float:
        """Measure memory usage of optimized code."""
        try:
            # Get baseline memory
            baseline = psutil.Process().memory_info().rss

            # Execute code
            exec(code)

            # Get new memory usage
            current = psutil.Process().memory_info().rss

            # Calculate impact (lower is better)
            return min(1.0, (current - baseline) / (1024 * 1024))  # MB scale

        except Exception:
            return 1.0

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature for similarity comparison."""
        return (
            f"{len(node.args.args)}:{len(node.body)}:{self._get_complexity_score(node)}"
        )

    def _get_complexity_score(self, node: ast.AST) -> int:
        """Calculate code complexity score."""
        complexity = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
        return complexity
