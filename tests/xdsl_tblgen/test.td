// Tablegen file used to generate `test.json`, which can be generated with the following command:
// ```
// llvm-tblgen -D test test.td -I $LLVM/mlir/include --dump-json | xdsl-tblgen -c > test.json
// ```
// where $LLVM is the root directory of an LLVM repository.

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

class Test_Type<string name, string typeMnemonic, list<Trait> traits = []>
: TypeDef<Test_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Test_Attr<string name, string attrMnemonic> : AttrDef<Test_Dialect, name> {
  let mnemonic = attrMnemonic;
}

class Test_Op<string mnemonic, list<Trait> traits = []>
    : Op<Test_Dialect, mnemonic, traits>;

def Test_SingletonAType : Test_Type<"SingletonAType", "singleton_a"> {}
def Test_SingletonBType : Test_Type<"SingletonBType", "singleton_b"> {}
def Test_SingletonCType : Test_Type<"SingletonCType", "singleton_c"> {}
def Test_TestAttr : Test_Attr<"Test", "test"> {}


// Check that AllOfType is converted correctly.
def Test_AndOp : Test_Op<"and"> {
  let arguments = (ins AllOfType<[Test_SingletonAType, AnyType]>:$in);
}

// Check that AnyType is converted correctly.
def Test_AnyOp : Test_Op<"any"> {
  let arguments = (ins AnyType:$in);
}

// Check assembly format is converted correctly.
def Test_AssemblyFormat : Test_Op<"assembly"> {
  let assemblyFormat = "attr-dict";
}

// Check assembly format is converted correctly.
def Test_AssemblyFormatLong : Test_Op<"assembly_long"> {
  let assemblyFormat = [{
      `a`
      `very`
      `long`
      `assembly`
      `format`
      attr-dict
  }];
}

// Check attributes are converted correctly.
def Test_AttributesOp : Test_Op<"attributes"> {
  let arguments = (ins I16Attr:$int_attr,
                       Test_TestAttr:$in,
                       OptionalAttr<Test_TestAttr>:$opt);
}

// Check confined types are converted correctly.
def Test_ConfinedOp : Test_Op<"confined"> {
  let arguments = (ins ConfinedType<AnyType, [CPred<"::llvm::isa<::mlir::TensorType>($_self)">]>:$tensor,
                       ConfinedType<AnyType, [And<[CPred<"::llvm::isa<::mlir::VectorType>($_self)">
                                             , CPred<"::llvm::cast<::mlir::VectorType>($_self).getRank() > 0">]>]>:$vector);
}

// Check generic integer types are converted correctly.
def Test_Integers : Test_Op<"integers"> {
  let arguments = (ins AnyI8:$any_int,
                       AnyInteger:$any_integer);
}

// Check that AnyTypeOf is converted correctly.
def Test_OrOp : Test_Op<"or"> {
  let arguments = (ins AnyTypeOf<[Test_SingletonAType, Test_SingletonBType, Test_SingletonCType]>:$in);
}

// Check that an op with a summary has a docstring
def Test_SummaryOp: Test_Op<"summary"> {
  let summary = "Op with a summary.";
}

// Check that various types are converted correctly.
def Test_TypesOp : Test_Op<"types"> {
  let arguments = (ins I32:$a,
                       SI64:$b,
                       UI8:$c,
                       Index:$d,
                       F32:$e,
                       NoneType:$f,
                       SymbolNameAttr:$g,
                       Complex<F8E4M3FN>);
}

// Check that variadics and optionals are converted correctly.
def Test_VariadicityOp : Test_Op<"variadicity"> {
  let arguments = (ins Variadic<Test_SingletonAType>:$variadic,
                       Optional<Test_SingletonBType>:$optional,
                       Test_SingletonCType:$required);
}

// Check that the AnyOf attribute is converted correctly.
def Test_AnyAttrOfOp : Test_Op<"any_attr_of_i32_str"> {
  let arguments = (ins AnyAttrOf<[I32Attr, StrAttr]>:$attr);
}

// Check that the AnyOf attribute with only one attr is converted correctly.
def Test_AnyAttrOfSingleOp : Test_Op<"any_attr_of_i32"> {
  let arguments = (ins AnyAttrOf<[I32Attr]>:$attr);
}
