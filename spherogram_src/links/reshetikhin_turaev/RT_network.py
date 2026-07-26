from itertools import product as cartesian_product

from .sparse_array import SparseTensor


class _ShapeOnly:
    """Stand-in for a tensor when only its shape matters (opt_einsum input)."""

    __slots__ = ["shape"]

    def __init__(self, shape):
        self.shape = shape


class DirectedEdge:
    __slots__ = ["label", "index", "sign", "reversed_edge"]

    def __init__(self, label, reversed_edge=None):
        self.label = label
        self.index = max(label, ~label)
        self.sign = 1 if label == self.index else -1

        if reversed_edge is None:
            reversed_edge = DirectedEdge(~self.label, self)
        self.reversed_edge = reversed_edge

    def __str__(self):
        return ("" if self.sign == 1 else "~") + str(self.index)

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return self.label == other.label

    def __invert__(self):
        return self.reversed_edge


class RTNetwork:

    def __init__(
        self,
        tensors,
        T=None,
        network=None,
        rot_num=None,
        boundary=None,
        boundary_labels=None,
    ):
        """
        Represent the tensor network obtained by applying
        the Reshetikhin--Turaev functor determined by the
        given RMatrix tensors to the tangle T.

        The network is represented as a list of pairs (tensor, legs)

        Requires numpy and opt_einsum modules for finding out the optimal contraction sequences
        """
        self.tensors = tensors

        # Whether the edge tensors are diagonal.  Slicing works either way,
        # but a diagonal h makes it cheapest; see choose_slice_edges.
        self.diagonal_h = tensors.diagonal_h if tensors is not None else False

        if T is not None:
            assert (
                T.is_upward()
            ), "Tangle should be upward for the Reshetikhin--Turaev functor to apply"
            self.rot_num = T.rot_num()
            self.tangle = T.copy()

            self.boundary = T.boundary
            self.boundary_labels = T.strand_labels

            self.idle_labels = set(self.boundary_labels)

            self.edge = edge = dict()

            self.network = network = []
            for c in T.crossings:
                labels = c.strand_labels
                for lab in labels:
                    if lab not in edge.keys():
                        edge[lab] = DirectedEdge(lab)

                if c.sign == 1:
                    key = (
                        ~edge[labels[3]],
                        ~edge[labels[0]],
                        edge[labels[2]],
                        edge[labels[1]],
                    )
                else:
                    assert c.sign == -1, f"Crossing {c} is not oriented"
                    key = (
                        ~edge[labels[0]],
                        ~edge[labels[1]],
                        edge[labels[3]],
                        edge[labels[2]],
                    )

                if tensors is not None:
                    tensor = tensors.R(c.sign)
                    for i, e in enumerate(list(key[:2])):
                        if e.index in self.idle_labels and self.rot_num[e.index] != 0:
                            perm = list(range(i)) + [3] + list(range(i, 3))
                            # h_ref tensors are shared, so this contraction must
                            # not consume them.
                            tensor = tensor.decorated_contract(
                                tensors.h_ref(0),
                                {(i, 1): (1, tensors.h_ref(self.rot_num[e.index]))},
                            )
                            tensor = tensor.permute(perm)

                    network.append((tensor, key))
                else:
                    network.append((None, key))

            for arc in self.idle_labels:
                if arc not in edge.keys():
                    edge[arc] = DirectedEdge(arc)

                    if tensors is not None:
                        network.append((tensors.h(0), (~edge[arc], edge[arc])))
                    else:
                        network.append((None, (~edge[arc], edge[arc])))
        else:
            assert all(
                item is not None
                for item in (network, rot_num, boundary, boundary_labels)
            )
            self.network = network
            self.rot_num = rot_num
            self.boundary = boundary
            self.boundary_labels = boundary_labels
            self.idle_labels = set(boundary_labels)

            self.edge = edge = dict()
            for _, key in network:
                for e in key:
                    if e.index not in edge.keys():
                        edge[e.index] = e if e.sign == 1 else ~e

    def __eq__(self, other):
        if len(self.network) != 1 or len(other.network) != 1:
            raise NotImplementedError(
                "Equality is only implemented for contracted networks"
            )
        else:
            return self.network[0][0] == other.network[0][0]

    def optimal_contraction_sequence(self):
        try:
            import opt_einsum as oe
        except ImportError:
            raise ModuleNotFoundError(
                "Module opt_einsum is required for computing the optimal contraction sequences"
            )

        oe_network = []
        idle = set(self.idle_labels)
        for tensor, key in self.network:
            if len(key) == 2 and key[0].index == key[1].index:
                try:
                    idle.remove(key[0].index)
                except:
                    raise ValueError(f"key {key[0].index} not found in {idle}")
            else:
                shape = tensor.shape if tensor is not None else tuple(8 for _ in key)
                oe_network.append(_ShapeOnly(shape))
                oe_network.append([edge.index for edge in key])

        return oe.contract_path(*oe_network, idle)[0]

    @staticmethod
    def local_contraction_width(abstract_network, indices):
        idx1, idx2 = indices
        ans = list(abstract_network)
        key1 = abstract_network[idx1]
        key2 = abstract_network[idx2]

        contracted_indices = set()

        pairs = []
        for pos_i, ei in enumerate(key1):
            for pos_j, ej in enumerate(key2):
                if ei.index == ej.index and ei.sign * ej.sign == -1:
                    pairs.append((pos_i, pos_j))
                    contracted_indices.add(ei.index)

        contracted1 = {pos_i for pos_i, _ in pairs}
        contracted2 = {pos_j for _, pos_j in pairs}

        if idx1 == idx2:
            contracted_all = contracted1 | contracted2
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted_all
            )
            ans.pop(idx1)
        else:
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted1
            ) + tuple(e for pos, e in enumerate(key2) if pos not in contracted2)
            hi, lo = max(idx1, idx2), min(idx1, idx2)
            ans.pop(hi)
            ans.pop(lo)

        ans.append(new_key)

        return len(new_key) + len(contracted_indices), ans

    def seq_contraction_width(self, seq):
        abstract_network = [key for _, key in self.network]

        w = 0
        m = 0

        for indices in seq:
            local_width, abstract_network = RTNetwork.local_contraction_width(
                abstract_network, indices
            )

            if local_width > w:
                w = local_width
                m = 1
            elif local_width == w:
                m += 1

        return (w, m), abstract_network

    def contraction_width(self, omit_idle_arcs=True):
        abstract_network = []
        for _, key in self.network:
            if omit_idle_arcs:
                non_idle_key = tuple(e for e in key if e.index not in self.idle_labels)
                abstract_network.append((None, non_idle_key))
            else:
                abstract_network.append((None, key))

        abstract_copy = RTNetwork(
            None,
            network=abstract_network,
            rot_num=self.rot_num,
            boundary=(0, 0) if omit_idle_arcs else self.boundary,
            boundary_labels=[] if omit_idle_arcs else self.boundary_labels,
        )

        loops = abstract_copy._resolve_self_loops()
        if loops:
            if all(len(loop) > 1 for loop in loops):
                width = (2, len(loops))
            else:
                width = (3, len([loop for loop in loops if len(loop) == 1]))
        else:
            width = (0, 0)

        seq_width, ans = abstract_copy.seq_contraction_width(
            abstract_copy.optimal_contraction_sequence()
        )

        return max(width, seq_width), ans

    @staticmethod
    def local_contraction_seq(abstract_network, indices):
        idx1, idx2 = indices
        ans = list(abstract_network)
        key1 = abstract_network[idx1]
        key2 = abstract_network[idx2]

        contracted_indices = set()

        pairs = []
        for pos_i, ei in enumerate(key1):
            for pos_j, ej in enumerate(key2):
                if ei.index == ej.index and ei.sign * ej.sign == -1:
                    pairs.append((pos_i, pos_j))
                    contracted_indices.add(ei.index)

        contracted1 = {pos_i for pos_i, _ in pairs}
        contracted2 = {pos_j for _, pos_j in pairs}

        if idx1 == idx2:
            contracted_all = contracted1 | contracted2
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted_all
            )
            ans.pop(idx1)
        else:
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted1
            ) + tuple(e for pos, e in enumerate(key2) if pos not in contracted2)
            hi, lo = max(idx1, idx2), min(idx1, idx2)
            ans.pop(hi)
            ans.pop(lo)

        ans.append(new_key)

        return contracted_indices, ans

    def seq_contraction_seq(self, seq):
        abstract_network = [key for _, key in self.network]

        ans = []

        for indices in seq:
            contracted_indices, abstract_network = RTNetwork.local_contraction_seq(
                abstract_network, indices
            )

            ans.append(contracted_indices)

        return ans, abstract_network

    def contraction_sequence(self, omit_idle_arcs=True):
        abstract_network = []
        for _, key in self.network:
            if omit_idle_arcs:
                non_idle_key = tuple(e for e in key if e.index not in self.idle_labels)
                abstract_network.append((None, non_idle_key))
            else:
                abstract_network.append((None, key))

        abstract_copy = RTNetwork(
            None,
            network=abstract_network,
            rot_num=self.rot_num,
            boundary=(0, 0) if omit_idle_arcs else self.boundary,
            boundary_labels=[] if omit_idle_arcs else self.boundary_labels,
        )

        loops = abstract_copy._resolve_self_loops()

        contraction_seq, ans = abstract_copy.seq_contraction_seq(
            abstract_copy.optimal_contraction_sequence()
        )

        return loops + contraction_seq, ans

    def contract_nodes(self, indices):
        """
        This modifies self to avoid holding duplicate data in memory.
        """
        idx1, idx2 = indices
        tensor1, key1 = self.network[idx1]
        tensor2, key2 = self.network[idx2]

        pairs = {}
        for pos_i, ei in enumerate(key1):
            for pos_j, ej in enumerate(key2):
                if ei.index == ej.index and ei.sign * ej.sign == -1:
                    side = 0 if ei.sign == 1 else 1
                    decoration = (
                        self.tensors.h_ref(self.rot_num[ei.index])
                        if self.tensors is not None
                        else None
                    )
                    pairs[(pos_i, pos_j)] = (side, decoration)

        if tensor1 is not None:
            # Both operands leave the network below, so let the contraction
            # cannibalise whichever one it streams.
            result_tensor = tensor1.decorated_contract(tensor2, pairs, consume=True)
        else:
            result_tensor = None

        contracted1 = {pos_i for pos_i, _ in pairs}
        contracted2 = {pos_j for _, pos_j in pairs}

        if idx1 == idx2:
            contracted_all = contracted1 | contracted2
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted_all
            )
            self.network.pop(idx1)
        else:
            new_key = tuple(
                e for pos, e in enumerate(key1) if pos not in contracted1
            ) + tuple(e for pos, e in enumerate(key2) if pos not in contracted2)
            hi, lo = max(idx1, idx2), min(idx1, idx2)
            self.network.pop(hi)
            self.network.pop(lo)

        del tensor1, tensor2

        self.network.append((result_tensor, new_key))

    def _resolve_self_loop_at(self, idx):
        _, key = self.network[idx]
        edge_indices = []
        seen = {}
        for pos, e in enumerate(key):
            if e.index in seen:
                _, other_e = seen[e.index]
                if e.sign * other_e.sign == -1:
                    edge_indices.append(e.index)
                    break
            else:
                seen[e.index] = (pos, e)
        if edge_indices:
            self.contract_nodes((idx, idx))
        return edge_indices

    def _resolve_self_loops(self):
        ans = []
        i = 0
        # Resolving a loop at i removes that node and appends its contraction
        # at the end, so the next node shifts into slot i and must be examined
        # before advancing.  A node may also carry more than one self-loop,
        # since _resolve_self_loop_at resolves a single one per call.
        while i < len(self.network):
            loop = self._resolve_self_loop_at(i)
            if loop:
                ans.append(loop)
            else:
                i += 1
        return ans

    def contract_sequence(self, seq, timed=False):
        if timed:
            import time

            start_time = time.time()

        for indices in seq:
            self.contract_nodes(indices)

        if timed:
            time_cost = time.time() - start_time

            return time_cost

    def _reorder_boundary(self):
        """
        Permute the single remaining tensor into the canonical boundary order.

        No-op for a network without boundary labels.
        """
        if not self.boundary_labels:
            return

        desired_order = []
        for i in range(self.boundary[0]):
            desired_order.append(~self.edge[self.boundary_labels[i]])
        for i in range(self.boundary[1]):
            desired_order.append(self.edge[self.boundary_labels[self.boundary[0] + i]])

        # Drop the network's reference first so the permuted copy and the
        # original never both hold the whole tensor.
        tensor, key = self.network.pop()
        key_pos = {e: i for i, e in enumerate(key)}
        reshape_indices = [key_pos[e] for e in desired_order]

        self.network.append(
            (tensor.permute(reshape_indices, consume=True), tuple(desired_order))
        )

    def largest_intermediate(self, cut=()):
        """
        Dense size of the largest intermediate tensor an optimal contraction
        of self would build, ignoring the edges whose indices are in `cut`.

        This is the quantity that determines peak memory, so it is what
        choose_slice_edges minimises.
        """
        try:
            import opt_einsum as oe
        except ImportError:
            raise ModuleNotFoundError(
                "Module opt_einsum is required for computing the optimal contraction sequences"
            )

        cut = set(cut)
        oe_network = []
        idle = set(self.idle_labels)
        for tensor, key in self.network:
            if len(key) == 2 and key[0].index == key[1].index:
                idle.discard(key[0].index)
                continue
            shape = tensor.shape if tensor is not None else tuple(8 for _ in key)
            oe_network.append(
                _ShapeOnly(
                    tuple(s for s, e in zip(shape, key) if e.index not in cut)
                )
            )
            oe_network.append([e.index for e in key if e.index not in cut])

        _, info = oe.contract_path(*oe_network, sorted(idle - cut))
        return int(info.largest_intermediate)

    def _slice_candidates(self):
        """Indices of the internal edges of self, which are the sliceable ones."""
        counts = {}
        for _, key in self.network:
            for e in key:
                counts[e.index] = counts.get(e.index, 0) + 1
        return sorted(
            index
            for index, count in counts.items()
            if count == 2 and index not in self.idle_labels
        )

    def _slice_count(self, index):
        """How many slices cutting the edge `index` costs: the nonzero entries
        of its edge tensor."""
        return len(self.tensors.h_ref(self.rot_num[index]))

    def choose_slice_edges(self, max_edges=2, min_factor=None, min_size=10**4):
        """
        Greedily choose edges to slice, or [] when slicing is not worth it.

        Cutting an edge divides the peak memory by roughly the factor by which
        it shrinks the largest intermediate, and multiplies the number of
        contractions by the number of slices it costs -- the nonzero entries
        of its edge tensor, which is the dimension when h is diagonal and up
        to its square when it is not.  So an edge is accepted only when
        cutting it divides the largest intermediate by at least that many,
        i.e. when it at least breaks even.  Slicing an edge that does not
        reduce the width costs a large multiple of the time for almost no
        memory, which is why the default is to refuse rather than to slice
        something.

        max_edges: most edges to cut.
        min_factor: override the break-even factor for every candidate.
        min_size: never slice a contraction whose largest intermediate is
                already below this, where the overhead would dominate.
        """
        if self.tensors is None:
            return []

        candidates = self._slice_candidates()
        if not candidates:
            return []

        current = self.largest_intermediate()
        if current < min_size:
            return []

        chosen = []
        while len(chosen) < max_edges and candidates:
            best, best_size = None, None
            for candidate in candidates:
                size = self.largest_intermediate(chosen + [candidate])
                factor = (
                    min_factor if min_factor is not None
                    else self._slice_count(candidate)
                )
                if size * factor > current:
                    # Cutting this edge costs more contractions than the
                    # memory it saves is worth.
                    continue
                if best_size is None or size < best_size:
                    best, best_size = candidate, size
            if best is None:
                break
            chosen.append(best)
            candidates.remove(best)
            current = best_size
            if current < min_size:
                break
        return chosen

    def _contract_all_sliced(self, slice_edges):
        """
        Contract self by cutting `slice_edges` and summing over their values.

        A cut edge joins a positively signed leg to a negatively signed one
        through its edge tensor h, contributing sum_{i,j} h[i,j] * (...) with
        i the value at the positive end and j the value at the negative end.
        Pinning those two ends to a fixed (i, j) and contracting the smaller
        network that remains, then summing the results weighted by h[i,j],
        gives the same answer while every intermediate carries one axis less.

        The slices run over the nonzero entries of h, so a cut costs as many
        contractions as h has entries: the dimension when h is diagonal (the
        two ends then always take the same value), up to its square when not.
        """
        # For each cut edge: its two legs with the value each takes, and its
        # edge tensor.
        sites = []
        for index in slice_edges:
            legs = [
                (node_pos, axis_pos, e.sign)
                for node_pos, (_, key) in enumerate(self.network)
                for axis_pos, e in enumerate(key)
                if e.index == index
            ]
            assert len(legs) == 2, f"edge {index} does not join exactly two legs"
            assert (
                legs[0][2] * legs[1][2] == -1
            ), f"edge {index} does not join opposite ends"
            sites.append((legs, self.tensors.h_ref(self.rot_num[index])))

        accumulator = None
        result_key = None
        seq = None

        for combination in cartesian_product(*[sorted(h.keys()) for _, h in sites]):
            weight = 1
            fixings = {}
            for (i, j), (legs, h) in zip(combination, sites):
                weight = weight * h[i, j]
                for node_pos, axis_pos, sign in legs:
                    # h[i,j] pairs the positive end's value with the negative
                    # end's; they coincide exactly when h is diagonal.
                    fixings.setdefault(node_pos, []).append(
                        (axis_pos, i if sign == 1 else j)
                    )

            sliced = []
            for node_pos, (tensor, key) in enumerate(self.network):
                fix = fixings.get(node_pos)
                if fix:
                    new_key = list(key)
                    for axis_pos, value in sorted(fix, reverse=True):
                        tensor = tensor.fixate(axis_pos, value)
                        new_key.pop(axis_pos)
                    sliced.append((tensor, tuple(new_key)))
                else:
                    # Every slice reads these, so hand out a private copy the
                    # contraction is free to consume.
                    sliced.append((tensor.copy(), key))

            slice_network = RTNetwork(
                self.tensors,
                network=sliced,
                rot_num=self.rot_num,
                boundary=self.boundary,
                boundary_labels=self.boundary_labels,
            )
            slice_network._resolve_self_loops()
            # Every slice has the same index structure, so the contraction
            # sequence found for the first one serves for all of them.
            if seq is None or len(seq) != len(slice_network.network) - 1:
                seq = slice_network.optimal_contraction_sequence()
            slice_network.contract_sequence(seq)
            assert len(slice_network.network) == 1
            slice_network._reorder_boundary()

            tensor, result_key = slice_network.network[0]
            if accumulator is None:
                accumulator = SparseTensor(tensor.shape, default=tensor.default)
            # value * weight is a fresh object in every case, so the
            # accumulator may absorb it.
            for key, value in tensor.items():
                accumulator._accumulate_owned(key, value * weight)

        if accumulator is None:
            # No edge value contributed, which needs an h tensor that is
            # identically zero.  Fall back rather than guess the result shape.
            return self._contract_all_direct()

        self.network = [(accumulator, result_key)]

    def contract_all(self, timed=False, slice_edges=None, **slice_options):
        """
        Perform all possible contractions on self.

        Return modified self and time (None if not timed).

        The contraction is sliced when that is worth it: some edges are cut
        and summed over separately, which keeps every intermediate smaller.
        Slicing works for any edge tensors, but is cheapest with diagonal h
        (see RMatrix.diagonal_h), where a cut costs one contraction per
        dimension rather than one per nonzero entry of h.  Pass slice_edges to
        choose the edges yourself, slice_edges=[] to force the plain
        contraction, or keyword arguments for choose_slice_edges to tune the
        selection.
        """
        if timed:
            import time

            start_time = time.time()

        self._resolve_self_loops()

        if slice_edges is None:
            slice_edges = self.choose_slice_edges(**slice_options)
        elif slice_options:
            raise ValueError("slice options are only used when choosing slice_edges")

        if slice_edges:
            self._contract_all_sliced(slice_edges)
        else:
            self._contract_all_direct()

        return (self, time.time() - start_time if timed else None)

    def _contract_all_direct(self):
        """Contract self in one pass, without slicing."""
        self.contract_sequence(self.optimal_contraction_sequence())
        assert len(self.network) == 1
        self._reorder_boundary()

    def evaluate(self, timed=False, slice_edges=None, **slice_options):
        """
        Fixate all idle labels at value 0, obtaining a new RTNework with (0,0) boundary (without modifying self),
        contract_all on the new RTNetwork and return the product of all values of the resulting tensors.

        slice_edges and any further keyword arguments are passed on to
        contract_all, which slices the contraction when the tensors have
        diagonal h.
        """
        assert self.boundary == (1, 1)

        new_network = []
        prefactor = 1

        for tensor, key in self.network:
            idle_positions = sorted(
                [pos for pos, e in enumerate(key) if e.index in self.idle_labels],
                reverse=True,
            )
            non_idle_key = tuple(e for e in key if e.index not in self.idle_labels)
            t = tensor
            for pos in idle_positions:
                t = t.fixate(pos, 0)
            if t.rank == 0:
                prefactor *= t[()]
            else:
                new_network.append((t, non_idle_key))

        reduced = RTNetwork(
            self.tensors,
            network=new_network,
            rot_num=self.rot_num,
            boundary=(0, 0),
            boundary_labels=[],
        )
        time = reduced.contract_all(
            timed=timed, slice_edges=slice_edges, **slice_options
        )[1]

        result = prefactor
        for tensor, _ in reduced.network:
            result *= tensor[()]
        return (result, time)
