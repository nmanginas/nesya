import os
import nnf
import time
import json
import torch
import itertools
from problog.logic import list2term
from deepstochlog.term import Term, List
from deepproblog.engines import ExactEngine
from deepproblog.model import Term as DPTerm
from deepfa.utils import parse_sapienza_to_fa
from deepstochlog.model import DeepStochLogModel
from deepproblog.model import Model, Query, Constant
from deepproblog.network import Network as DPNetwork
from deepstochlog.network import Network, NetworkStore
from deepstochlog.context import Context, ContextualizedTerm
from nesya.driving.scalability.utils import deepfa_to_problog, deepfa_to_stochlog


results_dir = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
        "results",
    )
)

print("scalability data writen to result dir: {}".format(results_dir))
results = {}
skip_problog = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

formulae = [
    "G((tired | blocked) -> WX(!fast))",
    "G((!(fast) | !(gesture_ok)) -> WX((blocked | tired) U (fast | gesture_ok)))",
    "G((battery_low | night | blocked | tired) -> WX(!fast)) & G(fast & WX(fast) -> WX(WX(battery_low)))",
]

for formula in formulae:
    start_time = time.time()
    deepfa = parse_sapienza_to_fa(formula)
    deepfa_compilation_time = time.time() - start_time

    for sequence_length in (
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    ):
        stochlog_program = deepfa_to_stochlog(deepfa)
        problog_program = deepfa_to_problog(deepfa)

        networks = {
            symbol: torch.nn.Sequential(
                torch.nn.Linear(1, 2), torch.nn.Softmax(dim=-1)
            ).to(device)
            for symbol in deepfa.symbols
        }

        optimizer = torch.optim.Adam(
            itertools.chain.from_iterable(
                [network.parameters() for network in networks.values()]
            ),
            lr=1e-8,
        )

        sequence = torch.rand(sequence_length, 1).to(device)

        class DummyInputs:
            def __len__(self) -> int:
                return sequence.shape[0]

            def __getitem__(self, index):
                return sequence[int(index[0])]

        start_time = time.time()
        if not skip_problog:

            deepproblog_model = Model(
                problog_program,
                networks=[
                    DPNetwork(network, "{}_net".format(symbol))
                    for symbol, network in networks.items()
                ],
                load=False,
            )
            compilation_time = time.time() - start_time

            deepproblog_model.set_engine(ExactEngine(deepproblog_model))
            deepproblog_model.add_tensor_source("train", DummyInputs())

            start_time = time.time()

            deeproblog_prediction = deepproblog_model.solve(
                [
                    Query(
                        DPTerm(
                            "accept",
                            list2term(
                                [
                                    DPTerm("tensor", DPTerm("train", Constant(i)))
                                    for i in range(sequence_length)
                                ]
                            ),
                        )
                    )
                ],
            )
            list(deeproblog_prediction[0].result.values())[0].backward()
            evaluation_time = time.time() - start_time

            results.setdefault(formula, {}).setdefault("deepproblog", {})[
                sequence_length
            ] = {
                "acceptance_prob": list(deeproblog_prediction[0].result.values())[
                    0
                ].item(),
                "compilation_time": compilation_time,
                "evaluation_time": evaluation_time,
            }

        start_time = time.time()

        model = DeepStochLogModel.from_string(
            program_str=stochlog_program,
            networks=NetworkStore(
                *(
                    Network(
                        "{}_net".format(symbol),
                        network,
                        index_list=[Term("0"), Term("1")],
                    )
                    for symbol, network in networks.items()
                )
            ),
            query=Term(
                "accept",
                Term(str(sequence_length)),
                List(
                    *(Term("t{}".format(i)) for i in reversed(range(sequence_length)))
                ),
            ),
            verbose=False,
            device=device,
        )

        stochlog_compilation_time = time.time() - start_time

        dummy_terms = [
            ContextualizedTerm(
                context=Context(
                    {Term("t{}".format(i)): sequence[i] for i in range(sequence_length)}
                ),
                term=Term(
                    "accept",
                    Term(str(sequence_length)),
                    List(
                        *(
                            Term("t{}".format(i))
                            for i in reversed(range(sequence_length))
                        )
                    ),
                ),
                probability=int(torch.rand(1).item() > 0.5),
            )
        ] * 16

        start_time = time.time()
        optimizer.zero_grad()
        deepstochlog_prediction = model.predict_sum_product(dummy_terms)
        deepstochlog_prediction = deepstochlog_prediction.clip(1e-32, 1 - 1e-32)
        loss = torch.nn.functional.binary_cross_entropy(
            deepstochlog_prediction, torch.ones_like(deepstochlog_prediction)
        )
        loss.backward()
        optimizer.step()
        stochlog_evaluation_time = time.time() - start_time

        results.setdefault(formula, {}).setdefault("deepstochlog", {})[
            sequence_length
        ] = {
            "program": stochlog_program,
            "acceptance_prob": deepstochlog_prediction.tolist(),
            "compilation_time": stochlog_compilation_time,
            "evaluation_time": stochlog_evaluation_time,
        }

        weights = {
            symbol: network(sequence)[:, 1].unsqueeze(0).repeat(16, 1)
            for symbol, network in networks.items()
        }

        def labelling_function(var: nnf.Var) -> torch.Tensor:
            return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]

        start_time = time.time()
        optimizer.zero_grad()
        deepfa_prediction = deepfa.forward(labelling_function)
        deepfa_prediction = deepfa_prediction.clip(1e-32, 1 - 1e-32)
        loss = torch.nn.functional.binary_cross_entropy(
            deepfa_prediction, torch.ones_like(deepfa_prediction)
        )
        loss.backward()
        optimizer.step()

        results[formula].setdefault("nesya", {})[sequence_length] = {
            "acceptance_prob": deepfa_prediction.tolist(),
            "evaluation_time": time.time() - start_time,
        }

        with open(os.path.join(results_dir, "timing_results.json"), "w") as output_file:
            json.dump(results, output_file, indent=2)
