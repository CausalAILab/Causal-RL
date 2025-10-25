import itertools
import re
from collections import deque
from copy import deepcopy

'''Borrowed from CausalAILab/NCMCounterfactuals.'''
# causal_graph.py
class CausalGraph:
    def __init__(self, V, directed_edges=[], bidirected_edges=[]):
        self.de = directed_edges
        self.be = bidirected_edges

        self.v = list(V)
        self.set_v = set(V)
        self.pa = {v: set() for v in V}  # parents (directed edges)
        self.ch = {v: set() for v in V}  # children (directed edges)
        self.ne = {v: set() for v in V}  # neighbors (bidirected edges)
        self.bi = set(map(tuple, map(sorted, bidirected_edges)))  # bidirected edges

        for v1, v2 in directed_edges:
            self.pa[v2].add(v1)
            self.ch[v1].add(v2)

        for v1, v2 in bidirected_edges:
            self.ne[v1].add(v2)
            self.ne[v2].add(v1)
            self.bi.add(tuple(sorted((v1, v2))))

        self.pa = {v: sorted(self.pa[v]) for v in self.v}
        self.ch = {v: sorted(self.ch[v]) for v in self.v}
        self.ne = {v: sorted(self.ne[v]) for v in self.v}

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        self.cc = self._c_components()
        self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v}
        self.pap = {
            v: sorted(set(itertools.chain.from_iterable(
                self.pa[v2] + [v2]
                for v2 in self.v2cc[v]
                if self.v2i[v2] <= self.v2i[v])) - {v},
                      key=self.v2i.get)
            for v in self.v}
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)

    def subgraph(self, V_sub, V_cut_back=None, V_cut_front=None):
        assert V_sub.issubset(self.set_v)

        if V_cut_back is None:
            V_cut_back = set()
        if V_cut_front is None:
            V_cut_front = set()

        assert V_cut_back.issubset(self.set_v)
        assert V_cut_front.issubset(self.set_v)

        new_de = [(V1, V2) for V1, V2 in self.de
                  if V1 in V_sub and V2 in V_sub and V2 not in V_cut_back and V1 not in V_cut_front]
        new_be = [(V1, V2) for V1, V2 in self.be
                  if V1 in V_sub and V2 in V_sub and V1 not in V_cut_back and V2 not in V_cut_back]

        return CausalGraph(V_sub, new_de, new_be)

    def _sort(self):  # sort V topologically
        L = []
        marks = {v: 0 for v in self.v}

        def visit(v):
            if marks[v] == 2:
                return
            if marks[v] == 1:
                raise ValueError('Not a DAG.')

            marks[v] = 1
            for c in self.ch[v]:
                visit(c)
            marks[v] = 2
            L.append(v)

        for v in marks:
            if marks[v] == 0:
                visit(v)
        self.v = L[::-1]

    def _c_components(self):
        pool = set(self.v)
        cc = []
        while pool:
            cc.append({pool.pop()})
            while True:
                added = {k2 for k in cc[-1] for k2 in self.ne[k]}
                delta = added - cc[-1]
                cc[-1].update(delta)
                pool.difference_update(delta)
                if not delta:
                    break
        return [tuple(sorted(c, key=self.v2i.get)) for c in cc]

    def _maximal_cliques(self):
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.ne[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.ne[v]),
                              x.intersection(self.ne[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.ne[v]),
                          x.intersection(self.ne[v]))
            p.remove(v)
            x.add(v)

        return c2

    def ancestors(self, C):
        """
        Returns the ancestors of set C.
        """
        assert C.issubset(self.set_v)

        frontier = [c for c in C]
        an = {c for c in C}
        while len(frontier) > 0:
            cur_v = frontier.pop(0)
            for par_v in self.pa[cur_v]:
                if par_v not in an:
                    an.add(par_v)
                    frontier.append(par_v)

        return an

    def convert_set_to_sorted(self, C):
        return [v for v in self.v if v in C]

    def serialize(self, C):
        return tuple(self.convert_set_to_sorted(C))

    @classmethod
    def read(cls, filename):
        with open(filename) as file:
            mode = None
            V = []
            directed_edges = []
            bidirected_edges = []
            try:
                for i, line in enumerate(map(str.strip, file), 1):
                    if line == '':
                        continue

                    m = re.match('<([A-Z]+)>', line)
                    if m:
                        mode = m.groups()[0]
                        continue

                    if mode == 'NODES':
                        if line.isidentifier():
                            V.append(line)
                        else:
                            raise ValueError('invalid identifier')
                    elif mode == 'EDGES':
                        if '<->' in line:
                            v1, v2 = map(str.strip, line.split('<->'))
                            bidirected_edges.append((v1, v2))
                        elif '->' in line:
                            v1, v2 = map(str.strip, line.split('->'))
                            directed_edges.append((v1, v2))
                        else:
                            raise ValueError('invalid edge type')
                    else:
                        raise ValueError('unknown mode')
            except Exception as e:
                raise ValueError(f'Error parsing line {i}: {e}: {line}')
            return cls(V, directed_edges, bidirected_edges)

    def save(self, filename):
        with open(filename, 'w') as file:
            lines = ["<NODES>\n"]
            for V in self.v:
                lines.append("{}\n".format(V))
            lines.append("\n")
            lines.append("<EDGES>\n")
            for V1, V2 in self.de:
                lines.append("{} -> {}\n".format(V1, V2))
            for V1, V2 in self.be:
                lines.append("{} <-> {}\n".format(V1, V2))
            file.writelines(lines)

# symbolic_id_tools.py
class Punit:
    def __init__(self, V_set, do_set=None):
        self.V_set = V_set
        if do_set is not None:
            self.do_set = do_set
        else:
            self.do_set = set()

    def set_do(self, do_set):
        self.do_set = do_set
        for V in do_set:
            if V in self.V_set:
                self.V_set.remove(V)

        if len(self.V_set) == 0:
            return False
        return True

    def _marg_remove(self, V):
        if V in self.V_set:
            self.V_set.remove(V)

    def _marg_check_remove(self, V):
        if V in self.V_set:
            return True
        return False

    def _marg_check_contains(self, V):
        if V in self.V_set:
            return 1, 0
        return 0, 0

    def get_latex(self):
        return str(self)

    def __str__(self):
        out = "P("
        for V in self.V_set:
            out = out + V + ','
        out = out[:-1]
        if len(self.do_set) > 0:
            out = out + ' | do('
            for V in self.do_set:
                out = out + V + ','
            out = out[:-1] + ')'
        out = out + ')'
        return out

class Pexpr:
    def __init__(self, upper, lower, marg_set):
        self.upper = upper
        self.lower = lower
        self.marg_set = marg_set

    def add_marg(self, marg_V):
        for V in marg_V:
            if self._marg_check_remove(V):
                self._marg_remove(V)
            else:
                self.marg_set.add(V)

    def set_do(self, do_set):
        for term in self.upper:
            success = term.set_do(do_set)
            if not success:
                self.upper.remove(term)

        for term in self.lower:
            success = term.set_do(do_set)
            if not success:
                self.lower.remove(term)

        if len(self.upper) == 0:
            return False
        return True

    def _marg_remove(self, V):
        for Pu in self.upper:
            Pu._marg_remove(V)

    def _marg_check_remove(self, V):
        upper_count, lower_count = self._marg_check_contains(V)
        if lower_count == 0 and upper_count <= 1:
            return True
        return False

    def _marg_check_contains(self, V):
        upper_count = 0
        lower_count = 0
        for Pu in self.upper:
            up, low = Pu._marg_check_contains(V)
            upper_count += up
            lower_count += low
        for Pl in self.lower:
            up, low = Pl._marg_check_contains(V)
            lower_count += up + low
        return upper_count, lower_count

    def get_latex(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return self.upper[0].get_latex()

        out = "\\left["
        if len(self.marg_set) > 0:
            out = "\\sum_{"
            for M in self.marg_set:
                out = out + str(M) + ','
            out = out[:-1] + '}\\left['

        if len(self.lower) > 0:
            out = out + "\\frac{"
            for P in self.upper:
                out = out + P.get_latex()
            out = out + "}{"
            for P in self.lower:
                out = out + P.get_latex()
            out = out + "}"
        else:
            for P in self.upper:
                out = out + P.get_latex()

        out = out + '\\right]'
        return out

    def __str__(self):
        if len(self.marg_set) == 0 and len(self.lower) == 0 and len(self.upper) == 1:
            return str(self.upper[0])

        out = "["
        if len(self.marg_set) > 0:
            out = "sum{"
            for M in self.marg_set:
                out = out + str(M) + ','
            out = out[:-1] + '}['
        for P in self.upper:
            out = out + str(P)
        if len(self.lower) > 0:
            out = out + " / "
            for P in self.lower:
                out = out + str(P)
        out = out + ']'
        return out

def identify(X, Y, G):
    """
    Takes sets of variables X and Y as input.
    If identifiable, returns P(Y | do(X)) in the form of a Pexpr object.
    Otherwise, returns FAIL.
    """
    Q_evals = dict()
    V_eval = Pexpr(upper=[Punit(G.set_v)], lower=[], marg_set=set())
    Q_evals[tuple(G.v)] = V_eval

    raw_C = G.set_v.difference(X)
    an_Y = G.subgraph(raw_C).ancestors(Y)
    marg = an_Y.difference(Y)

    Q_list = G.cc

    Qy_list = G.subgraph(an_Y).cc
    if len(Qy_list) == 1:
        Qy = set(Qy_list[0])
        for raw_Q in Q_list:
            Q = set(raw_Q)
            if Qy.issubset(Q):
                _evaluate_Q(Q, G.set_v, Q_evals, G)
                result = _identify_help(Qy, Q, Q_evals, G)
                if result == "FAIL":
                    return "FAIL"
                result.add_marg(marg)
                return result
    else:
        upper = []
        for raw_Qy in Qy_list:
            Qy = set(raw_Qy)
            for raw_Q in Q_list:
                Q = set(raw_Q)
                if Qy.issubset(Q):
                    _evaluate_Q(Q, G.set_v, Q_evals, G)
                    result = _identify_help(Qy, Q, Q_evals, G)
                    if result == "FAIL":
                        return "FAIL"
                    upper.append(result)

        result = Pexpr(upper=upper, lower=[], marg_set=set())
        result.add_marg(marg)
        return result


def _identify_help(C, T, Q_evals, G):
    T_eval = Q_evals[G.serialize(T)]
    if C == T:
        return T_eval

    an_C = G.subgraph(T).ancestors(C)
    if an_C == T:
        return "FAIL"

    marg_out = T.difference(an_C)
    an_C_eval = deepcopy(T_eval)
    an_C_eval.add_marg(marg_out)
    Q_evals[G.serialize(an_C)] = an_C_eval

    Q_list = G.subgraph(an_C).cc
    for raw_Q in Q_list:
        Q = set(raw_Q)
        if C.issubset(Q):
            _evaluate_Q(Q, an_C, Q_evals, G)
            return _identify_help(C, Q, Q_evals, G)


def _evaluate_Q(A, B, Q_evals, G):
    """
    Given variable sets B and its subset A, with Q[B] stored in Q_evals, Q[A] is computed using Q[B] and
    stored in Q_evals.
    """
    assert A.issubset(B)
    assert B.issubset(G.set_v)

    A_key = G.serialize(A)
    if A_key in Q_evals:
        return

    A_list = G.convert_set_to_sorted(A)
    B_list = G.convert_set_to_sorted(B)
    B_eval = Q_evals[G.serialize(B)]

    upper = []
    lower = []

    start = 0
    i = 0
    j = 0
    while i < len(A_list):
        while A_list[i] != B_list[j]:
            j += 1
            start += 1

        while i < len(A_list) and A_list[i] == B_list[j]:
            i += 1
            j += 1

        up_term = deepcopy(B_eval)
        if j < len(B_list):
            up_term.add_marg(set(B_list[j:]))
        upper.append(up_term)
        if start != 0:
            low_term = deepcopy(B_eval)
            low_term.add_marg(set(B_list[start:]))
            lower.append(low_term)
        start = j

    Q_evals[A_key] = Pexpr(upper=upper, lower=lower, marg_set=set())