# This is file is taken from
# https://github.com/whitemech/grounding_LTLf_in_image_sequences
# with some minimal changes

import torch
from flloat.parser.ltlf import LTLfParser


def xor(a, b):
    return conjunction(disjunction(a, b), negation(conjunction(a, b)))


def not_final_states_loss(final_states, p_states):
    cum_and = 1.0
    for f in final_states:
        cum_and = conjunction(cum_and, negation(p_states[f]))
    return 1 - cum_and


def final_states_loss(final_states, p_states):
    cum_xor = 0.0
    for f in final_states:
        cum_xor = xor(cum_xor, p_states[f])
    return 1 - cum_xor


def conjunction(a, b):
    return a * b


def disjunction(a, b):
    return a + b - a * b


def implication(pre, post):
    return 1 - pre + pre * post


def negation(a):
    return 1 - a


def divide_args_n(guard):
    args = guard.split(",")
    args_str = []

    curr_arg_i = 0
    while curr_arg_i < len(args):
        arg0 = args[curr_arg_i]
        while arg0.count("(") != arg0.count(")"):
            curr_arg_i += 1
            arg0 = arg0 + "," + args[curr_arg_i]
        args_str.append(arg0)
        curr_arg_i += 1
    return args_str


def recursive_guard_evaluation(guard, action):
    if guard[0] == "a":
        guard = guard[4:-1]
        value = 1.0
        args = divide_args_n(guard)
        for arg in args:
            value = conjunction(value, recursive_guard_evaluation(arg, action))
        return value
    elif guard[0] == "o":
        guard = guard[3:-1]
        args = divide_args_n(guard)
        value = 0.0
        for arg in args:
            value = disjunction(value, recursive_guard_evaluation(arg, action))
        return value
    elif guard[0] == "n":
        guard = guard[4:-1]
        return negation(recursive_guard_evaluation(guard, action))
    elif guard[0] == "T":
        return 1.0
    else:
        sym = int(guard)
        return action[sym]


def recurrent_write_guard(guard):
    if str(type(guard)) == "And":
        args = list(guard._argset)
        string_g = "and("
        for arg in args:
            string_g = string_g + recurrent_write_guard(arg) + ","
        string_g = string_g[:-1]
        string_g += ")"
        return string_g
    if str(type(guard)) == "Or":
        args = list(guard._argset)
        string_g = "or("
        for arg in args:
            string_g = string_g + recurrent_write_guard(arg) + ","
        string_g = string_g[:-1]
        string_g += ")"
        return string_g
    if str(type(guard)) == "Not":
        arg = str(guard)[2:]

        return "not({})".format(arg)
    if str(type(guard)) == "<class 'sympy.core.symbol.Symbol'>":
        return str(guard)[1:]
    if str(type(guard)) == "<class 'sympy.logic.boolalg.BooleanTrue'>":
        return "T"
    else:
        print("Not recognized type for the guard: ", type(guard))
        assert 3 == 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UmiliFSA:
    def __init__(self, formula: str, number_of_symbols: int):
        parser = LTLfParser()
        parsed_formula = parser(formula)
        self.original_dfa = parsed_formula.to_automaton()
        self.numb_of_states = self.original_dfa._state_counter
        self.final_states = list(self.original_dfa._final_states)

        self.dfa = self.reduce_dfa_non_mutex()

        self.numb_of_symbols = number_of_symbols

    # input: sequence of symbols probabilities (N, num_of_symbols)
    def forward(self, symbols_prob):
        s = torch.zeros(self.numb_of_states)
        # initial state is 0 for construction
        s[0] = 1.0

        for action in symbols_prob:
            s = self.next_state(s, action)

        return s

    def next_state(self, state, action):
        nxt_stt = torch.zeros(state.size()).to(device)
        for s in self.dfa.keys():
            for sym in self.dfa[s].keys():
                action_guard = recursive_guard_evaluation(sym, action)
                vvv = conjunction(state[s], action_guard)

                nxt_stt[self.dfa[s][sym]] += vvv
        return nxt_stt

    def reduce_dfa_non_mutex(self):
        red_trans_funct = {}
        for s0 in self.original_dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.original_dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                label = recurrent_write_guard(label)

                red_trans_funct[s0][label] = key

        return red_trans_funct
