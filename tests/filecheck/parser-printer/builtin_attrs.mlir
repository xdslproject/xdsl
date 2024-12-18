// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    ^bb0(%arg0: index):
    "func.return"() : () -> ()
  }) {function_type = (index) -> (), sym_name = "index_type_func"} : () -> ()

  // CHECK: (index)

  "func.func"() ({}) {function_type = () -> (), sym_name = "unit_attr_func", unitarray = [unit]} : () -> ()

  // CHECK: "unitarray" = [unit]

  "func.func"() ({
    ^bb0(%arg0: i32, %arg1: i64, %arg2: i1):
    "func.return"() : () -> ()
  }) {function_type = (i32, i64, i1) -> (), sym_name = "int_type"} : () -> ()

  // CHECK: (i32, i64, i1)

  "func.func"() ({
    ^bb0(%arg0: si32, %arg1: si64, %arg2: si1):
    "func.return"() : () -> ()
  }) {function_type = (si32, si64, si1) -> (), sym_name = "signed_int_type"} : () -> ()

  // CHECK: (si32, si64, si1)

  "func.func"() ({
    ^bb0(%arg0: ui32, %arg1: ui64, %arg2: ui1):
    "func.return"() : () -> ()
  }) {function_type = (ui32, ui64, ui1) -> (), sym_name = "unsigned_int_type"} : () -> ()

  // CHECK: (ui32, ui64, ui1)

  "func.func"() ({
    ^bb0(%arg0: bf16, %arg1: f16, %arg2: f32, %arg3: f64, %arg4: f80, %arg5: f128):
    "func.return"() : () -> ()
  }) {function_type = (bf16, f16, f32, f64, f80, f128) -> (), sym_name = "float_type"} : () -> ()

  // CHECK: (bf16, f16, f32, f64, f80, f128)


  "func.func"() ({}) {function_type = () -> (), value = 42.0 : f32, sym_name = "float_attr"} : () -> ()

  // CHECK: 4.200000e+01 : f32

  "func.func"() ({}) {function_type = () -> (), value = true, sym_name = "true_attr"} : () -> ()

  // CHECK: "value" = true

  "func.func"() ({}) {function_type = () -> (), value = 1 : i1, sym_name = "true_explicit_attr"} : () -> ()

  // CHECK: "value" = true

  "func.func"() ({}) {function_type = () -> (), value = false, sym_name = "false_attr"} : () -> ()

  // CHECK: "value" = false

  "func.func"() ({}) {function_type = () -> (), value = 0 : i1, sym_name = "false_explicit_attr"} : () -> ()

  // CHECK: "value" = false


  "func.func"() ({}) {function_type = () -> (), value = 42 : i32, sym_name = "int_attr"} : () -> ()

  // CHECK: 42 : i32

  "func.func"() ({}) {function_type = () -> (), value = 54 : index, sym_name = "index_attr"} : () -> ()

  // CHECK: 54 : index

  "func.func"() ({}) {function_type = () -> (), value = 54 : f32, sym_name = "f32_attr"} : () -> ()

  // CHECK: 5.400000e+01 : f32

  "func.func"() ({}) {function_type = () -> (), value = 0x132 : i32, sym_name = "hex_int_attr"} : () -> ()

  // CHECK: 306 : i32

  "func.func"() ({}) {function_type = () -> (), value = 0x43990000 : f32, sym_name = "hex_f32_attr"} : () -> ()

  // CHECK: 3.060000e+02 : f32

  "func.func"() ({}) {function_type = () -> (), value = dense<"0xEEA7CC3DF47612BE2BA4173E8B75E8BDE0B915BDA3191CBE8388E0BDC826DB3DFE78273E6B037E3DEF140D3EF0B5803D4026693CD6B6E1BCE08B4DBDC3A9E63D943B163EE64E46BD808C253EB8F4893D30270CBE36696C3D045E1DBED06A703DA33EBBBD66D646BD36507BBD764D8FBD7010FA3DB6E1B53D9B83C8BDD33FA73D58AD293EB0A6123EAB2627BA40B4CB3C20E9B6BD805AB2BDE047BDBC809A743DE01ADD3D9B77D5BDCEE7043E00B8C1BDCBA80A3DBB03DA3D787C993D163968BC208510BDABFDB1BD8C07213EA34614BEAB06B73A0091413B8013B3BD768F193E7B6515BE7306833D363183BC36BC8B3CA016B7BD3E05D33DE67C28BDCABB0EBEDA2A013EA67DF6BD007EB5BA782A04BEAB69F73D16DD703D3B93A43D1BE45B3DEBAEE8BD8891F1BDF8B18F3D20EC923CE67101BE8382A8BDAB9EE7BA0006CA3AA3F224BE1B56A5BDC06B8A3DC3E6BE3D562310BB964B713C2CC11FBE4BC68F3DAEACD7BDFB093A3D00070F3EC3E4C93D5BCF0D3D1B01E13D9B7D7F3D537CD43D6BEDFBBC4BD9AEBD17BA023E569906BB86599CBD4E28073E1639F5BDF60909BE8B4727BEE4AD153EDF3C05BEB01913BEEB1A59BD03E8D4BD4BD3123D9EA381BD6058F03CD0EFF73D00747FBADBC5AEBD5054273E204DB4BD00CA683B1E28C93D3BCC2A3D9B0E683D4302923D9A3408BEABC89D3A565336BCC0A7F3BD76D1F93D68A3B93D44891C3E1685243E1B3FDBBD5E06A4BD2B4192BD2B19983C50C97B3D40A808BEC0994C3D4B3435BD0B88293D506749BDFC13063E2B7ADF3CF3B013BE"> : tensor<4x4x3x3xf32>, sym_name = "hex_f32_large_attr"} : () -> ()

  // CHECK: "value" = dense<[[[[0.0999296755, -0.143031895, 0.148087189], [-0.113505445, -0.0365542173, -0.152441546], [-0.109635375, 0.107007563, 0.163547486]], [[0.0620149784, 0.137775168, 0.0628470182], [0.0142303109, -0.0275530033, -0.0501822233], [0.112628482, 0.146711648, -0.0484150872]], [[0.161668777, 0.0673612952, -0.136868238], [0.0577175245, -0.153678954, 0.0586956143], [-0.0914280638, -0.04854431, -0.061355792]], [[-0.0699719638, 0.122101665, 0.0888094157], [-0.0979072675, 0.0816647038, 0.165700316], [0.143213987, -0.000637630641, 0.0248662233]]], [[[-0.0893118382, -0.0870866776, -0.0231055617], [0.0597176552, 0.107961416, -0.104232036], [0.129790515, -0.0945892334, 0.0338523798]], [[0.106452428, 0.0749444366, -0.0141737666], [-0.0352832079, -0.0869096145, 0.157255352], [-0.144800708, 0.00139637792, 0.00295358896]], [[-0.087439537, 0.149961323, -0.14589493], [0.0639771447, -0.0160146765, 0.0170575194], [-0.0893986225, 0.103037342, -0.0411347374]], [[-0.139388233, 0.126140028, -0.120356843], [-0.0013846755, -0.129068255, 0.120807014], [0.058804594, 0.0803589448, 0.0536843352]]], [[[-0.11361488, -0.11795336, 0.0701636672], [0.0179348588, -0.126411051, -0.0822801813], [-0.00176711881, 0.00154131651, -0.161081836]], [[-0.0807306394, 0.0675883293, 0.0932135805], [-0.00219937181, 0.0147274937, -0.15601033], [0.0702024326, -0.105309829, 0.0454196744]], [[0.13967514, 0.0985808596, 0.0346215777], [0.10986539, 0.0623756461, 0.103752755], [-0.0307528581, -0.0853753909, 1.276630e-01]], [[-0.00205381727, -0.0763426274, 0.131989688], [-0.119737789, -0.13382706, -0.163358852], [0.146171153, -0.130115017, -0.143652678]]], [[[-0.0530041866, -0.103958152, 0.0358460359], [-0.0633003563, 0.0293390155, 0.121062875], [-0.000974476337, -0.0853383169, 0.163407564]], [[-0.0880377293, 0.0035520792, 0.0982210487], [0.0416986756, 0.0566545539, 0.0712933764], [-0.133013159, 0.00120379531, -0.0111282673]], [[-0.118972301, 0.121981546, 0.0906437039], [0.152867377, 0.160663933, -0.107053958], [-0.0800902694, -0.0714133605, 0.0185666885]], [[0.0614712834, -0.133454323, 0.0499513149], [-0.0442393236, 0.0413895063, -0.0491707921], [0.130935609, 0.0272799339, -0.144229695]]]]> : tensor<4x4x3x3xf32>} : () -> ()

  "func.func"() ({}) {function_type = () -> (), value = "foo", sym_name = "string_attr"} : () -> ()

  // CHECK: "foo"


  "func.func"() ({}) {function_type = () -> (), value = [0, "foo"], sym_name = "list_attr"} : () -> ()

  // CHECK: [0 : i64, "foo"]


  "func.func"() ({
    ^bb0(%arg0: vector<4xf32>, %arg1: vector<f32>, %arg2: vector<1x12xi32>):
    "func.return"() : () -> ()
  }) {function_type = (vector<4xf32>, vector<f32>, vector<1x12xi32>) -> (), sym_name = "vector_type"} : () -> ()

  // CHECK: (vector<4xf32>, vector<f32>, vector<1x12xi32>)

  "func.func"() ({
    ^bb0(%arg0: vector<[4]xf32>, %arg1: vector<[4x4]xf32>, %arg2: vector<12x[2x3]xi32>):
    "func.return"() : () -> ()
  }) {function_type = (vector<[4]xf32>, vector<[4x4]xf32>, vector<12x[2x3]xi32>) -> (), sym_name = "nd_vector_type"} : () -> ()

  // CHECK: (vector<[4]xf32>, vector<[4x4]xf32>, vector<12x[2x3]xi32>)


  "func.func"() ({
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<1x12xi32>, %arg3: tensor<*xf64>, %arg4: tensor<0xi32>):
    "func.return"() : () -> ()
  }) {function_type = (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>, tensor<0xi32>) -> (), sym_name = "tensor_type"} : () -> ()

  // CHECK: (tensor<4xf32>, tensor<f32>, tensor<1x12xi32>, tensor<*xf64>, tensor<0xi32>)

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[[2, 3]]> : tensor<1x2xi32>,
                      value2 = dense<[0.0, 1.0]> : tensor<2xf64>,
                      sym_name = "dense_tensor_attr"} : () -> ()

  // CHECK{LITERAL}: "value1" = dense<[[2, 3]]> : tensor<1x2xi32>, "value2" = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<"0xFF00"> : tensor<2xi8>,
                      value2 = dense<"0xFF00FF00"> : tensor<1xi32>,
                      value3 = dense<"0xCAFEBABE"> : tensor<2xi32>,
                      sym_name = "dense_tensor_attr_hex"} : () -> ()
  // CHECK: "value1" = dense<[-1, 0]> : tensor<2xi8>, "value2" = dense<16711935> : tensor<1xi32>, "value3" = dense<-1095041334> : tensor<2xi32>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<"0xFF00CAFE"> : tensor<2x2xi8>,
                      sym_name = "dense_tensor_attr_hex_long"} : () -> ()
  // CHECK{LITERAL}: "value1" = dense<[[-1, 0], [-54, -2]]> : tensor<2x2xi8>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<"0xCAFEBABE"> : tensor<2xf32>,
                      value2 = dense<"0xCAFEBABEB00BAABE"> : tensor<1xf64>,
                      sym_name = "dense_tensor_attr_hex_float"} : () -> ()
  // CHECK: "value1" = dense<-0.365225136> : tensor<2xf32>, "value2" = dense<-7.7622132495927025e-07> : tensor<1xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[0]> : vector<1xi32>,
                      value2 = dense<[0.0, 1.0]> : vector<2xf64>,
                      sym_name = "dense_vector_attr"} : () -> ()

  // CHECK: "value1" = dense<0> : vector<1xi32>, "value2" = dense<[0.000000e+00, 1.000000e+00]> : vector<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<> : tensor<1x23x0x4xi32>,
                      value2 = dense<[[0.0], [1.0]]> : tensor<2x1xf64>,
                      sym_name = "dense_corner_attr"} : () -> ()

  // CHECK{LITERAL}: "value1" = dense<> : tensor<1x23x0x4xi32>, "value2" = dense<[[0.000000e+00], [1.000000e+00]]> : tensor<2x1xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<> : tensor<1x23x0x4xi32>,
                      value2 = dense<[[-0.0], [-1.0]]> : tensor<2x1xf64>,
                      sym_name = "dense_negative_attr"} : () -> ()

  // CHECK{LITERAL}: "value1" = dense<> : tensor<1x23x0x4xi32>, "value2" = dense<[[-0.000000e+00], [-1.000000e+00]]> : tensor<2x1xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<12> : tensor<2x3xi32>,
                      sym_name = "dense_trivial_attr"} : () -> ()

  // CHECK: "value1" = dense<12> : tensor<2x3xi32>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = dense<[true, false]> : tensor<2xi1>,
                      sym_name = "dense_bool_attr"} : () -> ()

  // CHECK: "value1" = dense<[true, false]> : tensor<2xi1>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = opaque<"test", "contents">,
                      value2 = opaque<"test", "contents"> : tensor<2xf64>,
                      sym_name = "opaque_attr"} : () -> ()

  // CHECK: "value1" = opaque<"test", "contents">, "value2" = opaque<"test", "contents"> : tensor<2xf64>

  "func.func"() ({}) {function_type = () -> (),
                      value = {"one"=1, "two"=2, "three"="three"},
                      sym_name = "dict_attr"} : () -> ()

  // CHECK: "one" = 1 : i64, "two" = 2 : i64, "three" = "three"

  "func.func"() ({}) {function_type = () -> (),
                      symbol = @some_symbol,
                      sym_name = "symbol_ref_attr"} : () -> ()

  // CHECK: "symbol" = @some_symbol

  "func.func"() ({}) {function_type = () -> (),
                      value1 = tensor<?xi32>,
                      sym_name = "non_static_shaped_tensor"} : () -> ()

  // CHECK: tensor<?xi32>

  "func.func"() ({}) {function_type = () -> (),
                      value1 = tensor<?xi32, "encoding">,
                      sym_name = "tensor_with_encoding"} : () -> ()

  // CHECK: tensor<?xi32, "encoding">

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<2xf32>,
                      sym_name = "fixed_memref"} : () -> ()

  // CHECK: memref<2xf32>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<f32>,
                      sym_name = "scalar_memref"} : () -> ()

  // CHECK: memref<f32>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<2x?xf32>,
                      sym_name = "semidynamic_memref"} : () -> ()

  // CHECK: memref<2x?xf32>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<2xf32, strided<[]>>,
                      sym_name = "strided_memref"} : () -> ()

  // CHECK: memref<2xf32, strided<[]>>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<2xf32, strided<[]>, 2>,
                      sym_name = "strided_memspace_memref"} : () -> ()

  // CHECK: memref<2xf32, strided<[]>, 2 : i64>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<2xf32, 2>,
                      sym_name = "memspace_memref"} : () -> ()

  // CHECK: memref<2xf32, 2 : i64>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<*xf32>,
                      sym_name = "dynamic_memref"} : () -> ()

  // CHECK: memref<*xf32>

  "func.func"() ({}) {function_type = () -> (),
                      memref = memref<*xf32, 4>,
                      sym_name = "dynamic_memspace_memref"} : () -> ()

  // CHECK: memref<*xf32, 4 : i64>


  "func.func"() ({}) {function_type = () -> (),
                      dense_resource = dense_resource<resource_1> : tensor<1xi32>,
                      sym_name = "dense_resource"} : () -> ()

  // CHECK: dense_resource<resource_1> : tensor<1xi32>

  "func.func"() ({}) {function_type = () -> (),
                      type_attr = index,
                      sym_name = "index_type"} : () -> ()

  // CHECK: "type_attr" = index

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[1, 0x23, -23, -0x21, ?], offset: -3>,
                      sym_name = "strided"} : () -> ()
  // CHECK: "strided" = strided<[1, 35, -23, -33, ?], offset: -3>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[], offset: ?>,
                      sym_name = "what_strided"} : () -> ()
  // CHECK: "strided" = strided<[], offset: ?>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[], offset: 0>,
                      sym_name = "trivial_strided"} : () -> ()
  // CHECK: "strided" = strided<[]>

  "func.func"() ({}) {function_type = () -> (),
                      strided = strided<[]>,
                      sym_name = "empty_strided"} : () -> ()
  // CHECK: "strided" = strided<[]>

  "func.func"() ({}) {function_type = () -> (),
                      complex = complex<i32>,
                      sym_name = "complex_i32"} : () -> ()
  // CHECK: "complex" = complex<i32>

  "func.func"() ({}) {function_type = () -> (),
                      complex = complex<f32>,
                      sym_name = "complex_f32"} : () -> ()
  // CHECK: "complex" = complex<f32>

  "func.func"() ({}) {function_type = () -> (),
                      function = () -> i32,
                      sym_name = "one_to_one_func"} : () -> ()
  // CHECK: "function" = () -> i32

  "func.func"() ({}) {function_type = () -> (),
                      function = (i1) -> (i32),
                      sym_name = "one_to_one_func_paren"} : () -> ()
  // CHECK: "function" = (i1) -> i32

  "func.func"() ({}) {function_type = () -> (),
                      function = (i1, i2) -> (i32, i64),
                      sym_name = "two_to_two_func"} : () -> ()
  // CHECK: "function" = (i1, i2) -> (i32, i64)

  "func.func"() ({}) {function_type = () -> (),
                      function = () -> (() -> i32),
                      sym_name = "higher_order_func"} : () -> ()
  // CHECK: "function" = () -> (() -> i32)

}) : () -> ()
