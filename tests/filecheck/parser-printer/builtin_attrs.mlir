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

  // CHECK: "value" = dense<[[[[0.09992967545986176, -0.14303189516067505, 0.14808718860149384], [-0.11350544542074203, -0.03655421733856201, -0.15244154632091522], [-0.1096353754401207, 0.10700756311416626, 0.16354748606681824]], [[0.062014978379011154, 0.13777516782283783, 0.06284701824188232], [0.014230310916900635, -0.027553003281354904, -0.050182223320007324], [0.11262848228216171, 0.14671164751052856, -0.048415087163448334]], [[0.1616687774658203, 0.06736129522323608, -0.13686823844909668], [0.05771752446889877, -0.15367895364761353, 0.05869561433792114], [-0.09142806380987167, -0.04854431003332138, -0.06135579198598862]], [[-0.069971963763237, 0.12210166454315186, 0.08880941569805145], [-0.09790726751089096, 0.08166470378637314, 0.16570031642913818], [0.14321398735046387, -0.0006376306409947574, 0.024866223335266113]]], [[[-0.08931183815002441, -0.08708667755126953, -0.02310556173324585], [0.059717655181884766, 0.10796141624450684, -0.10423203557729721], [0.1297905147075653, -0.0945892333984375, 0.03385237976908684]], [[0.10645242780447006, 0.07494443655014038, -0.01417376659810543], [-0.03528320789337158, -0.08690961450338364, 0.1572553515434265], [-0.14480070769786835, 0.0013963779201731086, 0.0029535889625549316]], [[-0.08743953704833984, 0.14996132254600525, -0.14589492976665497], [0.06397714465856552, -0.01601467654109001, 0.017057519406080246], [-0.08939862251281738, 0.10303734242916107, -0.04113473743200302]], [[-0.13938823342323303, 0.1261400282382965, -0.12035684287548065], [-0.0013846755027770996, -0.1290682554244995, 0.12080701440572739], [0.05880459398031235, 0.08035894483327866, 0.053684335201978683]]], [[[-0.11361487954854965, -0.117953360080719, 0.07016366720199585], [0.017934858798980713, -0.1264110505580902, -0.08228018134832382], [-0.0017671188106760383, 0.0015413165092468262, -0.16108183562755585]], [[-0.08073063939809799, 0.06758832931518555, 0.09321358054876328], [-0.002199371811002493, 0.01472749374806881, -0.15601032972335815], [0.07020243257284164, -0.10530982911586761, 0.04541967436671257]], [[0.13967514038085938, 0.09858085960149765, 0.034621577709913254], [0.10986538976430893, 0.06237564608454704, 0.1037527546286583], [-0.03075285814702511, -0.08537539094686508, 0.1276630014181137]], [[-0.002053817268460989, -0.0763426274061203, 0.13198968768119812], [-0.11973778903484344, -0.1338270604610443, -0.16335885226726532], [0.14617115259170532, -0.13011501729488373, -0.14365267753601074]]], [[[-0.0530041866004467, -0.10395815223455429, 0.0358460359275341], [-0.06330035626888275, 0.0293390154838562, 0.12106287479400635], [-0.0009744763374328613, -0.08533831685781479, 0.163407564163208]], [[-0.08803772926330566, 0.003552079200744629, 0.09822104871273041], [0.041698675602674484, 0.05665455386042595, 0.07129337638616562], [-0.13301315903663635, 0.0012037953129038215, -0.011128267273306847]], [[-0.1189723014831543, 0.12198154628276825, 0.09064370393753052], [0.1528673768043518, 0.1606639325618744, -0.1070539578795433], [-0.08009026944637299, -0.07141336053609848, 0.018566688522696495]], [[0.06147128343582153, -0.1334543228149414, 0.04995131492614746], [-0.04423932358622551, 0.04138950631022453, -0.04917079210281372], [0.13093560934066772, 0.027279933914542198, -0.1442296952009201]]]]> : tensor<4x4x3x3xf32>

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
  // CHECK: "value1" = dense<-0.3652251362800598> : tensor<2xf32>, "value2" = dense<-7.762213249592702e-07> : tensor<1xf64>

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

  // CHECK: "value1" = dense<[1, 0]> : tensor<2xi1>

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
