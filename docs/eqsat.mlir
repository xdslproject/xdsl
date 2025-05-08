
// Idea eq sat in MLIR

// first example 2 * x = x << 1

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// Convert
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %res = arith.muli %x_eq, %c2_eq : index
    %res_eq = eqsat.eclass %res : index
    func.return %res_eq : index
}

// Add 2 * x = x << 1
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %c1 = arith.constant 1 : index
    %c1_eq = eqsat.eclass %c1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    %res = arith.muli %x_eq, %c2_eq : index
    %res_eq = eqsat.eclass %res, %shift : index
    func.return %res_eq : index
}

// Add 2 * x = x << 1 = x + x
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %c1 = arith.constant 1 : index
    %c1_eq = eqsat.eclass %c1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    %res = arith.muli %x_eq, %c2_eq : index
    %add = arith.add %x_eq, %x_eq : index
    %res_eq = eqsat.eclass %res, %shift, %add : index
    func.return %res_eq : index
}

// Add costs
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index {eqsat_cost=0}
    %c2 = arith.constant 2 attributes {eqsat_cost=1} : index
    %c2_eq = eqsat.eclass %c2 : index {eqsat_cost=1}
    %c1 = arith.constant 1 attributes {eqsat_cost=1} : index
    %c1_eq = eqsat.eclass %c1 : index {eqsat_cost=1}
    %shift = arith.shli %x_eq, %c1_eq attributes {eqsat_cost=3} : index  // add: lambda c0, c1: c0 + c1 + 2
    %res = arith.muli %x_eq, %c2_eq attributes {eqsat_cost=5} : index  // add: lambda c0, c1: c0 + c1 + 4
    %add = arith.add %x_eq, %x_eq attributes {eqsat_cost=3} : index  // add: lambda c0, c1: c0 + c1 + 2
    %res_eq = eqsat.eclass %res, %shift, %add attributes {eqsat_cost=3}: index
    func.return %res_eq : index
}

// Extract: minimal area (* = 10, + = 1, << = 0)
func.func @test(%x : index) -> (index) {
    %c1 = arith.constant 1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    func.return %res : index
}
