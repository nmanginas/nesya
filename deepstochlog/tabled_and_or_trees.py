import itertools
from functools import lru_cache
from typing import Dict, Tuple, List, Callable, Iterable

import torch
from torch import Tensor

from deepstochlog.context import Context, ContextualizedTerm
from deepstochlog.networkevaluation import NetworkEvaluations, RequiredEvaluation
from deepstochlog.term import Term
from deepstochlog.inferences import (
    NNLeafDescendantsRetriever,
    TermLeafDescendantsRetriever,
)


class TabledAndOrTrees:
    """
    Represents the grounded and/or tree, used for calculating the probabilities easily
    """

    def __init__(
        self,
        and_or_tree: Dict[Term, "LogicNode"],
        terms_grounder: Callable[[List[Term]], Dict[Term, "LogicNode"]] = None,
    ):
        self._and_or_tree = and_or_tree
        self._terms_grounder = terms_grounder

    def to_dot(self, term: Term):
        if term not in self._and_or_tree:
            raise RuntimeError("term {} is not in the tree".format(term))
        from deepstochlog.logic import Or, And, NNLeaf, StaticProbability, TermLeaf

        nodes, edges = [], []
        num_and_nodes = 0
        num_or_nodes = 0

        def recurse_term_node(term: Term, parent_hash):
            node = self._and_or_tree[term]
            node_hash = abs(hash(node) + parent_hash)
            if isinstance(node, Or):
                nonlocal num_or_nodes
                num_or_nodes += 1
                nodes.append(
                    "{} [label={}]".format(
                        node_hash,
                        '<OR<BR/> <FONT POINT-SIZE="7"> {} </FONT>>'.format(str(term)),
                    )
                )
            elif isinstance(node, And):
                nonlocal num_and_nodes
                num_and_nodes += 1
                nodes.append(
                    "{} [label={}]".format(
                        node_hash,
                        '<AND<BR/> <FONT POINT-SIZE="7"> {} </FONT>>'.format(str(term)),
                    )
                )

            elif isinstance(node, NNLeaf):
                nodes.append('{} [label="{}"]'.format(node_hash, str(node)))
                return

            for child in node.children:
                child_hash = abs(hash(child) + node_hash)
                if not isinstance(child, TermLeaf):
                    edges.append((node_hash, child_hash))
                resolve_other_node(child, node_hash)

        def resolve_other_node(node, parent_hash):
            node_hash = abs(hash(node) + parent_hash)
            if isinstance(node, TermLeaf):
                edges.append(
                    (
                        parent_hash,
                        abs(hash(self._and_or_tree[node.term]) + node_hash),
                    )
                )
                recurse_term_node(node.term, node_hash)
                return
            elif isinstance(node, Or):
                nonlocal num_or_nodes
                num_or_nodes += 1
                nodes.append("{} [label={}]".format(node_hash, '"OR"'))
            elif isinstance(node, And):
                nonlocal num_and_nodes
                num_and_nodes += 1
                nodes.append("{} [label={}]".format(node_hash, '"AND"'))
            elif isinstance(node, StaticProbability):
                nodes.append("{} [label={}]".format(node_hash, node.probability))
                return
            elif isinstance(node, NNLeaf):
                nodes.append('{} [label="{}"]'.format(node_hash, str(node)))
                return
            else:
                raise RuntimeError("unexpected node type")

            for child in node.children:
                child_hash = abs(hash(child) + node_hash)
                if not isinstance(child, TermLeaf):
                    edges.append((node_hash, child_hash))
                resolve_other_node(child, node_hash)

        recurse_term_node(term, 0)
        dot_string = (
            "Digraph {\n"
            + "\n".join(nodes)
            + "\n"
            + "\n".join(
                [
                    "{} -> {}".format(source, destination)
                    for source, destination in edges
                ]
            )
            + "\n}"
        )
        print(
            "run circuit with {} nodes and {} edges ({} and nodes, {} or nodes)".format(
                len(nodes), len(edges), num_and_nodes, num_or_nodes
            )
        )
        return dot_string, {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_or_nodes": num_or_nodes,
            "num_and_nodes": num_and_nodes,
        }

    def has_and_or_tree(self, term: Term):
        return term in self._and_or_tree

    def get_and_or_tree(self, term: Term):
        if not self.has_and_or_tree(term):
            self.ground_all([term])
        return self._and_or_tree[term]

    def ground_all(self, terms: List[Term]):
        if self._terms_grounder is None:
            raise RuntimeError(
                "Did not contain a term grounder, so could not ground missing terms",
                [str(term) for term in terms],
            )
        new_elements = self._terms_grounder(terms)
        self._and_or_tree.update(new_elements)

    def calculate_required_evaluations(
        self, contextualized_term: ContextualizedTerm
    ) -> List[RequiredEvaluation]:
        nn_leafs = list()
        handled_terms = set()
        term_queue = [contextualized_term.term]
        # Get all descendent nn_leafs
        while len(term_queue) > 0:
            t = term_queue.pop()
            if t not in handled_terms:
                handled_terms.add(t)
                tree = self.get_and_or_tree(t)
                # Extend with all neural leafs
                nn_leafs.extend(
                    tree.accept_visitor(visitor=NNLeafDescendantsRetriever())
                )
                # Add all term leafs as terms to handle
                term_queue.extend(
                    [
                        t.term
                        for t in tree.accept_visitor(
                            visitor=TermLeafDescendantsRetriever()
                        )
                    ]
                )

        # Turn every leaf into a required evaluation
        return [
            nn_leaf.to_required_evaluation(contextualized_term.context)
            for nn_leaf in nn_leafs
        ]


class LogicProbabilityEvaluator:
    def __init__(
        self,
        trees: TabledAndOrTrees,
        network_evaluations: NetworkEvaluations,
        device=None,
    ):
        self.device = device
        self.trees = trees
        self.network_evaluations = network_evaluations

    @lru_cache()
    def accept_term_visitor(self, term: Term, visitor: "LogicNodeVisitor"):
        return self.trees.get_and_or_tree(term).accept_visitor(visitor=visitor)

    def evaluate_neural_network_probability(
        self,
        network: str,
        input_arguments: Tuple,
        index: int,
        context: Context = None,
    ) -> Tensor:
        return self.network_evaluations.get_evaluation_result(
            context=context, network_name=network, input_args=input_arguments
        )[index]
