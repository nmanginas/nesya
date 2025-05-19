import nnf
import sympy
import string
from deepfa.automaton import DeepFA, nnf2str


def guard_to_lp(guard: nnf.NNF) -> list[str] | bool:
    dnf_formula = sympy.to_dnf(sympy.simplify(nnf2str(guard)))

    if isinstance(dnf_formula, sympy.logic.boolalg.BooleanTrue):
        return True
    if isinstance(dnf_formula, sympy.logic.boolalg.BooleanFalse):
        return False
    if isinstance(dnf_formula, sympy.Symbol):
        return ["{}(X, 1)".format(str(dnf_formula))]
    if isinstance(dnf_formula, sympy.logic.boolalg.Not):
        return ["{}(X, 0)".format(str(dnf_formula.args[0]))]

    def get_clause_from_and(
        conjunction: sympy.logic.boolalg.And | sympy.Symbol | sympy.logic.boolalg.Not,
    ) -> str:

        if isinstance(conjunction, sympy.Symbol):
            return "{}(X, 1)".format(str(conjunction))
        if isinstance(conjunction, sympy.logic.boolalg.Not):
            return "{}(X, 0)".format(str(conjunction.args[0]))

        return ", ".join(
            [
                (
                    "{}(X, 0)".format(str(arg.args[0]))
                    if isinstance(arg, sympy.logic.boolalg.Not)
                    else "{}(X, 1)".format(str(arg))
                )
                for arg in conjunction.args
            ]
        )

    return list(
        map(
            get_clause_from_and,
            (
                [dnf_formula]  # type: ignore
                if isinstance(dnf_formula, sympy.logic.boolalg.And)
                else dnf_formula.args
            ),
        )
    )


def guard_to_lp_stochlog(guard: nnf.NNF, symbols: list[str]) -> list[str] | bool:
    dnf_formula = sympy.to_dnf(sympy.simplify(nnf2str(guard)))
    uppercase_vars = dict(zip(symbols, string.ascii_uppercase))

    if isinstance(dnf_formula, sympy.logic.boolalg.BooleanTrue):
        return True
    if isinstance(dnf_formula, sympy.logic.boolalg.BooleanFalse):
        return False
    if isinstance(dnf_formula, sympy.Symbol):
        return ["{} is 1".format(uppercase_vars[str(dnf_formula)])]
    if isinstance(dnf_formula, sympy.logic.boolalg.Not):
        return ["{} is 0".format(uppercase_vars[str(dnf_formula.args[0])])]

    def get_clause_from_and(
        conjunction: sympy.logic.boolalg.And | sympy.Symbol | sympy.logic.boolalg.Not,
    ) -> str:

        if isinstance(conjunction, sympy.Symbol):
            return "{} is 1".format(uppercase_vars[str(conjunction)])
        if isinstance(conjunction, sympy.logic.boolalg.Not):
            return "{} is 0".format(uppercase_vars[str(conjunction.args[0])])

        return ", ".join(
            [
                (
                    "{} is 0".format(uppercase_vars[str(arg.args[0])])
                    if isinstance(arg, sympy.logic.boolalg.Not)
                    else "{} is 1".format(uppercase_vars[str(arg)])
                )
                for arg in conjunction.args
            ]
        )

    return list(
        map(
            get_clause_from_and,
            (
                [dnf_formula]  # type: ignore
                if isinstance(dnf_formula, sympy.logic.boolalg.And)
                else dnf_formula.args
            ),
        )
    )


def deepfa_to_problog(deepfa: DeepFA) -> str:
    program = []
    program.extend(
        [
            "split_last([X], [], X).",
            "split_last([H|T], [H|Rest], Last) :- split_last(T, Rest, Last).",
        ]
    )
    for symbol in deepfa.symbols:
        program.append("nn({0}_net, [X], Y, [0, 1]) :: {0}(X, Y).".format(symbol))

    for state in deepfa.states:
        for destination, guard in deepfa.transitions[state].items():
            transition_str = "transition({}, {}, X)".format(state, destination)
            disjuncts = guard_to_lp(guard)
            if disjuncts is True:
                program.append(transition_str + ".")
            elif disjuncts is False:
                continue
            else:
                for disjunct in disjuncts:
                    program.append(transition_str + " :- " + disjunct + ".")

    program.extend(
        [
            "state([], {}).".format(deepfa.initial_state),
            "state(S, Y) :- split_last(S, H, T), state(H, X), transition(X, Y, T).",
        ]
    )
    for state in deepfa.accepting_states:
        program.append("accept(S) :- state(S, {}).".format(state))

    return "\n".join(program)


def deepfa_to_stochlog(deepfa: DeepFA) -> str:
    program = []
    program.append("binary_dom(Y) :- member(Y, [0, 1]).")
    for symbol in deepfa.symbols:
        program.append(
            "nn({0}_net, [X], Y, binary_dom) :: {0}(X, Y) --> [].".format(symbol)
        )

    symbols = list(deepfa.symbols)

    for state in deepfa.states:
        for destination, guard in deepfa.transitions[state].items():
            transition_str = "transition({}, {}, {})".format(
                state, destination, ", ".join(string.ascii_uppercase[: len(symbols)])
            )
            disjuncts = guard_to_lp_stochlog(guard, symbols)
            if disjuncts is True:
                program.append(transition_str + ".")
            elif disjuncts is False:
                continue
            else:
                for disjunct in disjuncts:
                    program.append(transition_str + " :- " + disjunct + ".")

    program.extend(
        [
            "state(0, 0) --> [].",
            "state(Y, T) --> {{ T > 0, T1 is T - 1 }}, [O], state(X, T1), {}, {{ transition(X, Y, {}) }}.".format(
                ", ".join(
                    [
                        "{}(O, {})".format(symbol, var)
                        for symbol, var in zip(symbols, string.ascii_uppercase)
                    ]
                ),
                ", ".join(string.ascii_uppercase[: len(symbols)]),
            ),
        ]
    )
    for state in deepfa.accepting_states:
        program.append("accept(T) --> state({}, T).".format(state))

    return "\n".join(program)
