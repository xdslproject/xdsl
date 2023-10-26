"""
This is a stub of CIRCT’s hw dialect.
It currently implements minimal types and operations used by other dialects.

[1] https://circt.llvm.org/docs/Dialects/HW/
"""

from xdsl.dialects.builtin import (
    FlatSymbolRefAttr,
    ParameterDef,
    StringAttr,
)
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)


class FieldIDTypeInterface:
    """Common methods for types which can be indexed by a field_id.

    field_id is a depth-first numbering of the elements of a type. For example:
    ```
    struct a  /* 0 */ {
      int b; /* 1 */
      struct c /* 2 */ {
        int d; /* 3 */
      }
    }

    int e; /* 0 */
    ```
    """

    def get_max_field_id(self) -> int:
        """Get the maximum field ID for this type"""
        return 0

    def get_sub_type_by_field_id(self, field_id: int) -> tuple[type, int]:
        """Get the sub-type of a type for a field ID, and the subfield's ID. Strip
        off a single layer of this type and return the sub-type and a field ID
        targeting the same field, but rebased on the sub-type.

        The resultant type *may* not be a FieldIDTypeInterface if the resulting
        field_id is zero. This means that leaf types may be ground without
        implementing an interface. An empty aggregate will also appear as a zero."""
        if field_id == 0:
            return (type(self), 0)
        raise NotImplementedError()

    def project_to_child_field_id(self, field_id: int, index: int) -> tuple[int, bool]:
        """Returns the effective field id when treating the index field as the
        root of the type. Essentially maps a field_id to a field_id after a
        subfield op. Returns the new id and whether the id is in the given
        child."""
        ...

    def get_index_for_field_id(self, field_id: int) -> int:
        """Returns the index (e.g. struct or vector element) for a given
        field_id. This returns the containing index in the case that the
        field_id points to a child field of a field."""
        ...

    def get_field_id(self, index: int) -> int:
        """Return the field_id of a given index (e.g. struct or vector element).
        field ids start at 1, and are assigned to each field in a recursive
        depth-first walk of all elements. A field ID of 0 is used to reference
        the type itself."""
        ...

    def get_index_and_subfield_id(self, field_id: int) -> tuple[int, int]:
        """Find the index of the element that contains the given field_id. As well, rebase the field_id to the element."""
        ...


class InnerSymTarget:
    """The target of an inner symbol, the entity the symbol is a handle for."""

    def __init__(
        self,
        op: Operation | None = None,
        field_id: int = 0,
        port_idx: int | None = None,
    ) -> None:
        self.op = op
        self.field_id = field_id
        self.port_idx = port_idx

    def __bool__(self):
        # None-valued op defines an invalid target
        return self.op is not None

    def is_port(self) -> bool:
        return self.port_idx is not None

    def is_field(self) -> bool:
        return self.field_id != 0

    def is_op_only(self) -> bool:
        return not self.is_field() and not self.is_port()

    @classmethod
    def get_target_for_subfield(
        cls, base: "InnerSymTarget", field_id: int
    ) -> "InnerSymTarget":
        return cls(base.op, base.field_id + field_id, base.port_idx)


@irdl_attr_definition
class InnerRefAttr(ParametrizedAttribute):
    """This works like a symbol reference, but to a name inside a module.

    NB: the parse and print for AsmPrinter are not copied from CIRCT."""

    name = "hw.inner_name_ref"
    module_ref: ParameterDef[FlatSymbolRefAttr]
    # NB. upstream defines as “name” which clashes with Attribute.name
    sym_name: ParameterDef[StringAttr]

    def __init__(self, module: str | StringAttr, name: str | StringAttr) -> None:
        if isinstance(module, str):
            module = StringAttr(module)
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__([FlatSymbolRefAttr(module), name])

    @classmethod
    def get_from_operation(
        cls, op: Operation, sym_name: StringAttr, module_name: StringAttr
    ) -> "InnerRefAttr":
        """Get the InnerRefAttr for an operation and add the sym on it."""
        # NB: declared upstream, but no implementation to be found
        raise NotImplementedError

    def get_module(self) -> StringAttr:
        """Return the name of the referenced module."""
        return self.module_ref.root_reference


HW = Dialect(
    [],
    [
        InnerRefAttr,
    ],
)
