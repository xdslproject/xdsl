"builtin.module"() ({
  %0:3 = "test.op"() : () -> (vector<index>, vector<3xindex>, index)
  %1 = "vector.insertelement"(%0#2, %0#0) : (index, vector<index>) -> vector<index>
  %2 = "vector.insertelement"(%0#2, %0#1, %0#2) : (index, vector<3xindex>, index) -> vector<3xindex>
  %3 = "vector.extractelement"(%0#1, %0#2) : (vector<3xindex>, index) -> index
  %4 = "vector.extractelement"(%0#0) : (vector<index>) -> index
}) : () -> ()
