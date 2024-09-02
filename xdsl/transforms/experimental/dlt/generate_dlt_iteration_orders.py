import datetime
import itertools
import time
import typing

from scripts.visualiseDLT import IterationPlotter
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import SetAttr
from xdsl.transforms.experimental.dlt.iteration_map import IterationMap


def _make_nested_order(order: dlt.IterationOrder) -> dlt.IterationOrder:
    if isinstance(order, dlt.BodyIterationOrderAttr):
        return order
    elif isinstance(order, dlt.NestedIterationOrderAttr):
        child = _make_nested_order(order.child)
        return dlt.NestedIterationOrderAttr(order.extent_index, child)
    elif isinstance(order, dlt.NonZeroIterationOrderAttr):
        child = _make_nested_order(order.child)
        return dlt.NonZeroIterationOrderAttr(order.extent_indices, order.tensor_index, child)
    elif isinstance(order, dlt.AbstractIterationOrderAttr):
        child = _make_nested_order(order.child)
        for extent_idx in order.extent_indices:
            child = dlt.NestedIterationOrderAttr(extent_idx, child)
        return child


def _make_non_zero_nested_order(order: dlt.IterationOrder) -> dlt.IterationOrder:
    if isinstance(order, dlt.BodyIterationOrderAttr):
        return order
    elif isinstance(order, dlt.NestedIterationOrderAttr):
        child = _make_non_zero_nested_order(order.child)
        return dlt.NestedIterationOrderAttr(order.extent_index, child)
    elif isinstance(order, dlt.NonZeroIterationOrderAttr):
        child = _make_non_zero_nested_order(order.child)
        return dlt.NonZeroIterationOrderAttr(order.extent_indices, order.tensor_index, child)
    elif isinstance(order, dlt.AbstractIterationOrderAttr):
        child = _make_non_zero_nested_order(order.child)
        best_tensor_idx = None
        extents = set()
        for tensor_idx, tensor_extents in zip(order.non_zero_reducible_tensors, order.non_zero_reducible_tensor_extents):
            if len(tensor_extents) > len(extents):
                extents = set(tensor_extents)
                best_tensor_idx = tensor_idx
        if best_tensor_idx is None:
            for extent_idx in order.extent_indices:
                child = dlt.NestedIterationOrderAttr(extent_idx, child)
            return child
        else:
            left_over_extents = SetAttr(set(order.extent_indices)-extents)
            left_over_tensors = []
            left_over_tensor_extents = []
            for tensor_idx, tensor_extents in zip(order.non_zero_reducible_tensors,
                                                  order.non_zero_reducible_tensor_extents):
                if tensor_idx != best_tensor_idx:
                    left_over_tensors.append(tensor_idx)
                    left_over_tensor_extents.append(SetAttr([e for e in tensor_extents if e in left_over_extents]))
            left_over_tensors = ArrayAttr(left_over_tensors)
            left_over_tensor_extents = ArrayAttr(left_over_tensor_extents)
            if len(set(left_over_extents)) > 0:
                child = dlt.AbstractIterationOrderAttr(left_over_extents, left_over_tensors, left_over_tensor_extents, child)
            return dlt.NonZeroIterationOrderAttr(extents, best_tensor_idx, child)


class IterationMapping():

    def __init__(self, number: int, iter_map: dict[StringAttr, dlt.IterationOrder]):
        self.number = number
        self.keys = tuple(iter_map.keys())
        self.values = tuple(iter_map.values())

    def __eq__(self, other) -> bool:
        if not isinstance(other, IterationMapping):
            return False
        return self.keys == other.keys and self.values == other.values

    def __hash__(self):
        return hash((self.keys, self.values))

    def __getitem__(self, key):
        for i, k in enumerate(self.keys):
            if k == key:
                return self.values[i]

    def make_iter_dict(self) -> dict[StringAttr, dlt.IterationOrder]:
        return {k:v for k,v in zip(self.keys, self.values)}


class IterationGenerator:

    def __init__(self, iteration_map: IterationMap, plot_dir=None):
        self.iteration_map = iteration_map
        self.final_maps: set[IterationMapping] = set()
        self.plot_dir = plot_dir

        if self.plot_dir is not None:
            self.plotter = IterationPlotter(iteration_map.get_ops())

    def plot(self, mapping: IterationMapping):
        if self.plot_dir is None:
            return
        mapping_dict = mapping.make_iter_dict()
        name = f"mapping_{mapping.number}"
        if not any(order.is_abstract for order in mapping_dict.values()):
            name += "_final"
        self.plotter.plot(mapping_dict, plot_name=name, directory=self.plot_dir)

    def generate_mappings(self, take_first=0):
        print(f"{datetime.datetime.now()} : Generating iteration mappings for {len(self.iteration_map.iteration_ops)} iteration ops.")

        self.plot(IterationMapping(0, self.iteration_map.get_map()))
        start_time = time.time()
        idents = []
        abstract_orders = []
        reified_orderings = []
        order_counts = []
        for id, a_order in self.iteration_map.get_map().items():
            print(f"\tReifying {id.data}", end="")
            idents.append(id)
            abstract_orders.append(a_order)
            new_orders = self.reify(a_order)
            reified_orderings.append(new_orders)
            order_counts.append(len(new_orders))
            print(f" -> {len(new_orders)} new iteration orders")

        product = 1
        _ = [(product := (product * c)) for c in order_counts]

        print(f"\tNew iteration mappings count: {product}")
        mapping_num = 1

        val = f"{mapping_num}/{product}"
        print(f"\tCompiling new mapping: {val}", end="")
        chars = len(val)

        for orderings in itertools.product(*reified_orderings):
            val = f"{mapping_num}/{product}"
            print(("\b"*chars) + val, end="")
            chars = len(val)

            new_mapping = {id: order for id, order in zip(idents, orderings)}
            new_map = IterationMapping(mapping_num, new_mapping)
            self.plot(new_map)
            self.final_maps.add(new_map)
            if 0 < take_first <= len(self.final_maps):
                print("")
                print(f"{datetime.datetime.now()} : Exiting early with {len(self.final_maps)} mappings")
                return self.final_maps

            mapping_num += 1

        print("")

        print(f"{datetime.datetime.now()} : Generated {mapping_num-1} iteration mappings in {time.time() - start_time} seconds")
        return self.final_maps

    def reify(self, order: dlt.IterationOrder) -> list[dlt.IterationOrder]:
        if not order.is_abstract():
            return [order]
        elif isinstance(order, dlt.AbstractIterationOrderAttr):
            reified_orders = self.reify_abstract(order)
            return reified_orders
        else:
            reified_orders = []
            new_children = []
            for child in order.get_children():
                new_children.append(self.reify(child))
            for children in itertools.product(*new_children):
                reified_orders.append(order.from_children(children))
            return reified_orders

    def reify_abstract(self, order: dlt.AbstractIterationOrderAttr) -> list[dlt.IterationOrder]:
        reified_orders = []
        children = self.reify(order.child)
        for permutation in itertools.permutations(list(order.extent_indices)):
            for child in children:
                new_order = dlt.NestedIterationOrderAttr.generate_for(list(permutation), body=child)
                reified_orders.append(new_order)


        for tensor_idx, tensor_extents in zip(order.non_zero_reducible_tensors, order.non_zero_reducible_tensor_extents):
            left_over_extents = SetAttr(set(order.extent_indices)-set(tensor_extents))
            left_over_tensors = []
            left_over_tensor_extents = []
            for t_idx, t_extents in zip(order.non_zero_reducible_tensors,
                                                  order.non_zero_reducible_tensor_extents):
                if t_idx != tensor_idx:
                    left_over_tensors.append(t_idx)
                    left_over_tensor_extents.append(SetAttr([e for e in t_extents if e in left_over_extents]))
            left_over_tensors = ArrayAttr(left_over_tensors)
            left_over_tensor_extents = ArrayAttr(left_over_tensor_extents)
            if len(set(left_over_extents)) > 0:
                abstract_child = dlt.AbstractIterationOrderAttr(left_over_extents, left_over_tensors, left_over_tensor_extents, order.child)
                reified_orders.append(dlt.NonZeroIterationOrderAttr(tensor_extents, tensor_idx, abstract_child))
            else:
                reified_orders.append(dlt.NonZeroIterationOrderAttr(tensor_extents, tensor_idx, order.child))
        return reified_orders
