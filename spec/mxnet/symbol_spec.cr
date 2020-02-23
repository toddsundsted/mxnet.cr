require "json"
require "../spec_helper"

private macro random_spec_helper(random, *args)
  describe ".{{random}}" do
    it "returns an array of random numbers" do
      MXNet::Symbol.{{random}}({{*args}}, shape: 2).eval.first.shape.should eq([2])
      MXNet::Symbol.{{random}}({{*args}}, shape: [1, 2, 3]).eval.first.shape.should eq([1, 2, 3])
      MXNet::Symbol.{{random}}({{*args}}, shape: 1, dtype: :float32).eval.first.dtype.should eq(:float32)
    end

    it "names the new symbol" do
      a = MXNet::Symbol.{{random}}({{*args}}, name: "a")
      a.name.should eq("a")
    end
  end
end

private macro sample_spec_helper(sample, *args)
  describe ".{{sample}}" do
    syms = {
      {% for i in (0...args.size) %}
        MXNet::Symbol.var({{i.stringify}}),
      {% end %}
    }
    args = {
      {% for arg in (args) %}
        MXNet::NDArray.array({{arg}}),
      {% end %}
    }

    it "returns an array of sampled numbers" do
      MXNet::Symbol.{{sample}}(*syms, shape: 2).eval(*args).first.shape.should eq([2, 2])
      MXNet::Symbol.{{sample}}(*syms, shape: [1, 2, 3]).eval(*args).first.shape.should eq([2, 1, 2, 3])
      MXNet::Symbol.{{sample}}(*syms, shape: 1, dtype: :float32).eval(*args).first.dtype.should eq(:float32)
    end

    it "names the new symbol" do
      a = MXNet::Symbol.{{sample}}(*syms, name: "a")
      a.name.should eq("a")
    end
  end
end

struct Expression
  struct Node
    JSON.mapping(
      op: String,
      name: String,
      inputs: Array(Array(Int32))
    )

    def initialize(@op, @name, @inputs)
    end
  end

  struct Attrs
    JSON.mapping(
      mxnet_version: Tuple(String, Int32)
    )

    def initialize
      MXNet::Internal.libcall(
        MXGetVersion,
        out version
      )
      @mxnet_version = {"int", version}
    end
  end

  JSON.mapping(
    nodes: Array(Node),
    arg_nodes: Array(Int32),
    node_row_ptr: Array(Int32),
    heads: Array(Array(Int32)),
    attrs: Attrs
  )

  def initialize(@nodes, @arg_nodes, @node_row_ptr, @heads)
    @attrs = Attrs.new
  end
end

describe MXNet::Symbol do
  describe ".create_symbol" do
    it "removes nil arguments" do
      s = MXNet::Symbol.var("s")
      MXNet::Symbol.create_symbol("reshape", s, shape: nil)
    end

    it "removes nil arguments" do
      s = MXNet::Symbol.var("s")
      MXNet::Symbol.create_symbol("elemwise_add", s, s, nil)
    end

    it "flattens arguments" do
      s = MXNet::Symbol.var("s")
      MXNet::Symbol.create_symbol("add_n", [s, s], num_args: 2)
    end
  end

  describe ".var" do
    it "creates a symbolic variable" do
      MXNet::Symbol.var("s").should be_a(MXNet::Symbol)
    end

    it "sets the given attributes" do
      data = MXNet::Symbol.var("data", attr: {mood: "angry"})
      data.attr("mood").should eq("angry")
    end

    it "sets the shape as an attribute" do
      data = MXNet::Symbol.var("data", shape: [1, 2, 3])
      data.attr("__shape__").should eq("[1, 2, 3]")
    end

    it "sets the dtype as an attribute" do
      data = MXNet::Symbol.var("data", dtype: :float32)
      data.attr("__dtype__").should eq("0")
    end
  end

  describe ".zeros" do
    it "returns an array filled with all zeros" do
      MXNet::Symbol.zeros([1, 2]).eval.first.should eq(MXNet::NDArray.array([[0.0, 0.0]], dtype: :float32))
      MXNet::Symbol.zeros(2, dtype: :int32).eval.first.should eq(MXNet::NDArray.array([0, 0], dtype: :int32))
    end

    it "names the new symbol" do
      a = MXNet::Symbol.zeros(1, name: "a")
      a.name.should eq("a")
    end
  end

  describe ".ones" do
    it "returns an array filled with all ones" do
      MXNet::Symbol.ones([1, 2]).eval.first.should eq(MXNet::NDArray.array([[1.0, 1.0]], dtype: :float32))
      MXNet::Symbol.ones(2, dtype: :int64).eval.first.should eq(MXNet::NDArray.array([1, 1], dtype: :int64))
    end

    it "names the new symbol" do
      a = MXNet::Symbol.ones(1, name: "a")
      a.name.should eq("a")
    end
  end

  describe ".save" do
    temp = File.tempname("save")
    x = MXNet::Symbol.var("x")
    y = MXNet::Symbol.var("y")
    z = x + y
    it "saves a symbol" do
      MXNet::Symbol.save(temp, z)
      exp = Expression.from_json(File.read(temp))
      exp.nodes.should contain(Expression::Node.new("null", "x", [] of Array(Int32)))
      exp.nodes.should contain(Expression::Node.new("null", "y", [] of Array(Int32)))
      exp.nodes.should contain(Expression::Node.new("elemwise_add", z.name.not_nil!, [[0, 0, 0], [1, 0, 0]]))
      exp.arg_nodes.should eq([0, 1])
      exp.node_row_ptr.should eq([0, 1, 2, 3])
      exp.heads.should eq([[2, 0, 0]])
    end
  end

  describe ".load" do
    temp = File.tempname("load")
    it "loads a symbol" do
      x = Expression::Node.new("null", "x", [] of Array(Int32))
      y = Expression::Node.new("null", "y", [] of Array(Int32))
      plus = Expression::Node.new("elemwise_add", "_plus0", [[0, 0, 0], [1, 0, 0]])
      exp = Expression.new(
        [x, y, plus],
        [0, 1],
        [0, 1, 2, 3],
        [[2, 0, 0]]
      )
      File.open(temp, "w") do |file|
        file.puts(exp.to_json)
      end
      s = MXNet::Symbol.load(temp)
      s.list_arguments.should eq(["x", "y"])
      s.name.should eq("_plus0")
    end
  end

  random_spec_helper(random_uniform, 0.0, 1.0)
  random_spec_helper(random_normal, 0.0, 1.0)
  random_spec_helper(random_poisson, 1.0)
  random_spec_helper(random_exponential, 1.0)
  random_spec_helper(random_gamma, 1.0, 1.0)

  sample_spec_helper(sample_uniform, [0.0, 2.5], [1.0, 3.7])
  sample_spec_helper(sample_normal, [0.0, 2.5], [1.0, 3.7])
  sample_spec_helper(sample_poisson, [1.0, 8.5])
  sample_spec_helper(sample_exponential, [1.0, 8.5])
  sample_spec_helper(sample_gamma, [0.0, 2.5], [1.0, 0.7])
  sample_spec_helper(
    sample_multinomial,
    [[0.0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.0]]
  )

  describe "#name" do
    it "returns the name of the symbol" do
      MXNet::Symbol.var("foo").name.should eq("foo")
    end
  end

  describe "#list_attr" do
    it "returns the attributes of the symbol" do
      data = MXNet::Symbol.var("data", attr: {"mood" => "angry"})
      data.list_attr.should eq({"mood" => "angry"})
    end
  end

  describe "#attr_dict" do
    it "returns the attributes of the symbol and its children" do
      a = MXNet::Symbol.var("a", attr: {"a1" => "a2"})
      b = MXNet::Symbol.var("b", attr: {"b1" => "b2"})
      c = a + b
      c.attr_dict.should eq({"a" => {"a1" => "a2"}, "b" => {"b1" => "b2"}})
    end
  end

  describe "#list_arguments" do
    it "returns the arguments of the symbol" do
      MXNet::Symbol.var("foo").list_arguments.should eq(["foo"])
    end
  end

  describe "#list_outputs" do
    it "returns the outputs of the symbol" do
      MXNet::Symbol.var("foo").list_outputs.should eq(["foo"])
    end
  end

  describe "#list_auxiliary_states" do
    it "returns the auxiliary states of the symbol" do
      MXNet::Symbol.var("foo").list_auxiliary_states.should be_empty
    end
  end

  _i32 = [] of Array(Int32)

  describe "#infer_shape" do
    a = MXNet::Symbol.var("a")
    b = MXNet::Symbol.var("b")
    c = a + MXNet::Symbol.concat([b, b])

    it "infers shape positionally" do
      c.infer_shape([nil, [3, 3]]).should eq({[[3, 6], [3, 3]], [[3, 6]], _i32})
    end

    it "infers shape by name" do
      c.infer_shape({"b" => [3, 3]}).should eq({[[3, 6], [3, 3]], [[3, 6]], _i32})
    end
  end

  describe "#infer_shape_partial" do
    a = MXNet::Symbol::Ops._FullyConnected(MXNet::Symbol.var("a"), nil, nil, num_hidden: 128)
    b = MXNet::Symbol::Ops._FullyConnected(MXNet::Symbol.var("b"), nil, nil, num_hidden: 128)
    c = a + b

    it "infers shape positionally" do
      c.infer_shape_partial([[10, 64]]).should eq({[[10, 64], [128, 64], [128], _i32, _i32, _i32], [[10, 128]], _i32})
    end

    it "infers shape by name" do
      c.infer_shape_partial({"a" => [10, 64]}).should eq({[[10, 64], [128, 64], [128], _i32, _i32, _i32], [[10, 128]], _i32})
    end
  end

  _sym = [] of ::Symbol

  describe "#infer_dtype" do
    a = MXNet::Symbol.var("a")
    b = MXNet::Symbol.var("b")
    c = a + b

    it "infers dtype positionally" do
      c.infer_dtype([nil, :int32]).should eq({[:int32, :int32], [:int32], _sym})
    end

    it "infers dtype by name" do
      c.infer_dtype({"b" => :int32}).should eq({[:int32, :int32], [:int32], _sym})
    end
  end

  describe "#infer_dtype_partial" do
    a = MXNet::Symbol.var("a")
    b = MXNet::Symbol::Ops._cast(MXNet::Symbol.var("b"), dtype: :int32)
    c = a + b

    it "infers dtype positionally" do
      {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.5.0") >= 0 %}
        c.infer_dtype_partial([:int32]).should eq({[:int32, nil], [:int32], _sym})
      {% end %}
    end

    it "infers dtype by name" do
      {% if compare_versions(MXNet::Internal::MXNET_VERSION, "1.5.0") >= 0 %}
        c.infer_dtype_partial({"a" => :int32}).should eq({[:int32, nil], [:int32], _sym})
      {% end %}
    end
  end

  describe "#bind" do
    it "binds an array of arguments" do
      MXNet::Symbol.var("a").bind(args: [MXNet::NDArray.array([1])]).should be_a(MXNet::Executor)
    end
    it "binds a hash of arguments" do
      MXNet::Symbol.var("a").bind(args: {"a" => MXNet::NDArray.array([1])}).should be_a(MXNet::Executor)
    end
  end

  describe "#eval" do
    it "evaluates arguments" do
      MXNet::Symbol.var("a").eval(MXNet::NDArray.array([1])).to_a.should be_a(Array(MXNet::NDArray))
    end
    it "evaluates named arguments" do
      MXNet::Symbol.var("a").eval(a: MXNet::NDArray.array([1])).to_a.should be_a(Array(MXNet::NDArray))
    end
  end

  describe "#to_s" do
    it "pretty-prints the symbol" do
      MXNet::Symbol.var("foo").to_s.should eq("<Symbol foo>")
    end
  end

  args = {
    a: MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]]),
    b: MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]]),
    c: MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]),
    d: MXNet::NDArray.array([[1.0], [4.0], [9.0]]),
    e: MXNet::NDArray.array([[-1.0], [1.0]]),
    f: MXNet::NDArray.array([-2.1, -1.9, 1.5, 1.9, 2.1]),
    i: MXNet::NDArray.array([0.0, 1.0]),
    z: MXNet::NDArray.array([0.0])
  }

  describe "#+" do
    it "adds a scalar to an array" do
      a = MXNet::Symbol.var("a")
      (a + 2).eval(**args).first.should eq(MXNet::NDArray.array([[3.0, 4.0], [5.0, 6.0]]))
      (2 + a).eval(**args).first.should eq(MXNet::NDArray.array([[3.0, 4.0], [5.0, 6.0]]))
    end
    it "adds two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a + b).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
    context "unary" do
      it "leaves the array unchanged" do
        (+MXNet::Symbol.var("a")).eval(MXNet::NDArray.array([1.0, 2.0])).first.should eq(MXNet::NDArray.array([1.0, 2.0]))
        (+MXNet::Symbol.var("b")).eval(MXNet::NDArray.array([1, 2])).first.should eq(MXNet::NDArray.array([1, 2]))
      end
    end
  end

  describe "#-" do
    it "subtracts a scalar from an array" do
      a = MXNet::Symbol.var("a")
      (a - 2).eval(**args).first.should eq(MXNet::NDArray.array([[-1.0, 0.0], [1.0, 2.0]]))
      (2 - a).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [-1.0, -2.0]]))
    end
    it "subtracts two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a - b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
    context "unary" do
      it "negates the array" do
        (-MXNet::Symbol.var("a")).eval(MXNet::NDArray.array([1.0, 2.0])).first.should eq(MXNet::NDArray.array([-1.0, -2.0]))
        (-MXNet::Symbol.var("b")).eval(MXNet::NDArray.array([1, 2])).first.should eq(MXNet::NDArray.array([-1, -2]))
      end
    end
  end

  describe "#*" do
    it "multiplies an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a * 2).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 4.0], [6.0, 8.0]]))
      (2 * a).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 4.0], [6.0, 8.0]]))
    end
    it "multiplies two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a * b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 8.0], [3.0, 4.0]]))
    end
  end

  describe "#/" do
    it "divides an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a / 2).eval(**args).first.should eq(MXNet::NDArray.array([[0.5, 1.0], [1.5, 2.0]]))
      (2 / a).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 1.0], [(1.0/1.5), 0.5]]))
    end
    it "divides two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a / b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.5], [3.0, 4.0]]))
    end
  end

  describe "#**" do
    it "exponentiates an array by a scalar" do
      a = MXNet::Symbol.var("a")
      (a ** 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 4.0], [9.0, 16.0]]))
      (2 ** a).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 4.0], [8.0, 16.0]]))
    end
    it "exponentiates two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a ** b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 16.0], [3.0, 4.0]]))
    end
  end

  describe "#==" do
    it "performs element-wise equal with a scalar" do
      a = MXNet::Symbol.var("a")
      (a == 2).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
      (2 == a).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
    it "performs element-wise equal" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a == b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
  end

  describe "#!=" do
    it "performs element-wise not equal with a scalar" do
      a = MXNet::Symbol.var("a")
      (a != 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
      (2 != a).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
    it "performs element-wise not equal" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a != b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe "#>" do
    it "performs element-wise greater than with a scalar" do
      a = MXNet::Symbol.var("a")
      (a > 2).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
      (2 > a).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
    it "performs element-wise greater than" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a > b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe "#>=" do
    it "performs element-wise greater than or equal to with a scalar" do
      a = MXNet::Symbol.var("a")
      (a >= 2).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
      (2 >= a).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
    it "performs element-wise greater than or equal to" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a >= b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe "#<" do
    it "performs element-wise less than with a scalar" do
      a = MXNet::Symbol.var("a")
      (a < 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
      (2 < a).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
    it "performs element-wise less than" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a < b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe "#<=" do
    it "performs element-wise less than or equal to with a scalar" do
      a = MXNet::Symbol.var("a")
      (a <= 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
      (2 <= a).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
    it "performs element-wise less than or equal to" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      (a <= b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".abs" do
    it "computes the element-wise absolute value of the input" do
      e = MXNet::Symbol.var("e")
      e.abs.eval(**args).first.should eq(MXNet::NDArray.array([[1.0], [1.0]]))
    end
  end

  describe ".add_n" do
    it "adds arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      MXNet::Symbol.add_n([a, b]).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".arange" do
    it "returns evenly spaced values within a given interval" do
      MXNet::Symbol.arange(3).eval(**args).first.should eq(MXNet::NDArray.array([0.0, 1.0, 2.0], :float32))
      MXNet::Symbol.arange(2, 6).eval(**args).first.should eq(MXNet::NDArray.array([2.0, 3.0, 4.0, 5.0], :float32))
      MXNet::Symbol.arange(2, 6, step: 2).eval(**args).first.should eq(MXNet::NDArray.array([2.0, 4.0], :float32))
    end
  end

  describe ".broadcast_add" do
    it "adds two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_add(b).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".broadcast_div" do
    it "divides two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_div(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.5], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_equal" do
    it "performs element-wise equal" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_equal(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_greater" do
    it "performs element-wise greater than" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_greater(b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_greater_equal" do
    it "performs element-wise greater than or equal to" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_greater_equal(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_lesser" do
    it "performs element-wise less than" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_lesser(b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_lesser_equal" do
    it "performs element-wise less than or equal to" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_lesser_equal(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_maximum" do
    it "returns the maximum" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_maximum(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 4.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_minimum" do
    it "returns the minimum" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_minimum(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 2.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_minus" do
    it "subtracts two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_minus(b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
  end

  describe ".broadcast_mul" do
    it "multiplies two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_mul(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 8.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_not_equal" do
    it "performs element-wise not equal" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_not_equal(b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_plus" do
    it "adds two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_plus(b).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".broadcast_power" do
    it "exponentiates two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_power(b).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 16.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_sub" do
    it "subtracts two arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      a.broadcast_sub(b).eval(**args).first.should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
  end

  describe ".clip" do
    it "clips the values in an array" do
      c = MXNet::Symbol.var("c")
      c.clip(2.0, 7.0).eval(**args).first.should eq(MXNet::NDArray.array([[[2.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 7.0]]]))
    end
  end

  describe ".concat" do
    it "concatenates arrays" do
      a = MXNet::Symbol.var("a")
      b = MXNet::Symbol.var("b")
      MXNet::Symbol.concat([a, b]).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 2.0, 1.0, 4.0], [3.0, 4.0, 1.0, 1.0]]))
    end
  end

  describe ".dot" do
    it "calculates the dot product of two arrays" do
      a = MXNet::Symbol.var("a")
      x = a.reshape(shape: [2, 2])
      y = a.reshape(shape: 4).flip(axis: 0).reshape(shape: [2, 2])
      MXNet::Symbol.dot(x, y).eval(**args).first.should eq(MXNet::NDArray.array([[8.0, 5.0], [20.0, 13.0]]))
    end
  end

  describe "#expand_dims" do
    it "inserts a new axis into the input array" do
      c = MXNet::Symbol.var("c")
      c.expand_dims(axis: 1).eval(**args).first.shape.should eq([1, 1, 4, 2])
      c.expand_dims(1).eval(**args).first.shape.should eq([1, 1, 4, 2])
    end
  end

  describe "#flatten" do
    it "flattens the input array" do
      c = MXNet::Symbol.var("c")
      c.flatten.eval(**args).first.shape.should eq([1, 8])
    end
  end

  describe "#flip" do
    it "reverses the order of elements" do
      c = MXNet::Symbol.var("c")
      c.flip(axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[[7.0, 8.0], [5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]]))
      c.flip(axis: 2).eval(**args).first.should eq(MXNet::NDArray.array([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]]))
    end
  end

  describe "#floor" do
    it "returns floor of the input" do
      f = MXNet::Symbol.var("f")
      f.floor.eval(**args).first.should eq(MXNet::NDArray.array([-3.0, -2.0, 1.0, 1.0, 2.0]))
    end
  end

  describe "#log" do
    it "computes the natural logarithm" do
      a = MXNet::Symbol.var("a")
      a.log.eval(**args).first.should be_close(MXNet::NDArray.array([[0.0, 0.69314], [1.0986, 1.3862]]), 0.001)
    end
  end

  describe "#log_softmax" do
    it "computes the log softmax of the input" do
      a = MXNet::Symbol.var("a")
      a.log_softmax(axis: 0).eval(**args).first.should be_close(MXNet::NDArray.array([[-2.1269, -2.1269], [-0.1269, -0.1269]]), 0.05)
      a.log_softmax(axis: 1).eval(**args).first.should be_close(MXNet::NDArray.array([[-1.3133, -0.3133], [-1.3133, -0.3133]]), 0.05)
    end
  end

  describe "#max" do
    it "computes the max" do
      c = MXNet::Symbol.var("c")
      c.max(axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[7.0, 8.0]]))
      c.max(axis: 2).eval(**args).first.should eq(MXNet::NDArray.array([[2.0, 4.0, 6.0, 8.0]]))
      c.max.eval(**args).first.should eq(MXNet::NDArray.array([8.0]))
    end
  end

  describe "#mean" do
    it "computes the mean" do
      c = MXNet::Symbol.var("c")
      c.mean(axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[4.0, 5.0]]))
      c.mean(axis: 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.5, 3.5, 5.5, 7.5]]))
      c.mean.eval(**args).first.should eq(MXNet::NDArray.array([4.5]))
    end
  end

  describe "#min" do
    it "computes the min" do
      c = MXNet::Symbol.var("c")
      c.min(axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 2.0]]))
      c.min(axis: 2).eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 3.0, 5.0, 7.0]]))
      c.min.eval(**args).first.should eq(MXNet::NDArray.array([1.0]))
    end
  end

  describe "#norm" do
    it "returns the norm" do
      a = MXNet::Symbol.var("a")
      a.norm.eval(**args).first.should be_close(MXNet::NDArray.array([5.47]), 0.05)
    end
  end

  describe "#one_hot" do
    it "returns a one-hot array" do
      b = MXNet::Symbol.var("b")
      o = MXNet::NDArray.array([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype: :float32)
      b.reshape(shape: [-1]).one_hot(5).eval(**args).first.should eq(o)
    end
  end

  describe ".ones_like" do
    it "creates an array of the same shape filled with ones" do
      a = MXNet::Symbol.var("a")
      z = MXNet::Symbol.ones_like(a)
      z.eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe "#pick" do
    it "picks elements from an input array" do
      a = MXNet::Symbol.var("a")
      i = MXNet::Symbol.var("i")
      a.pick(i, axis: 0).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 4.0]))
    end
  end

  describe "#relu" do
    it "computes the rectified linear activation of the input" do
      e = MXNet::Symbol.var("e")
      e.relu.eval(**args).first.should eq(MXNet::NDArray.array([[0.0], [1.0]]))
    end
  end

  describe "#reshape" do
    it "reshapes the input array" do
      a = MXNet::Symbol.var("a")
      a.reshape(shape: [4]).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape(shape: 4).eval(**args).first.should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
    end

    it "supports special values for dimensions" do
      c = MXNet::Symbol.var("c")
      c.reshape(shape: [-1, 0], reverse: false).eval(**args).first.shape.should eq([2, 4])
      c.reshape(shape: [-1, 0], reverse: true).eval(**args).first.shape.should eq([4, 2])
    end
  end

  describe "#reshape_like" do
    it "reshapes the input array" do
      a = MXNet::Symbol.zeros(shape: [9])
      b = MXNet::Symbol.zeros(shape: [3, 3])
      a.reshape_like(b).eval.first.shape.should eq([3, 3])
    end
  end

  describe ".shuffle" do
    it "randomly shuffles the elements" do
      a = MXNet::Symbol.var("a")
      MXNet::Symbol.shuffle(a).eval(**args).first.should be_a(MXNet::NDArray)
    end
  end

  describe "#sigmoid" do
    it "computes the sigmoid activation of the input" do
      z = MXNet::Symbol.var("z")
      z.sigmoid.eval(**args).first.as_scalar.should eq(0.5)
    end
  end

  describe "#sign" do
    it "returns element-wise sign of the input" do
      e = MXNet::Symbol.var("e")
      e.sign.eval(**args).first.should eq(MXNet::NDArray.array([[-1.0], [1.0]]))
    end
  end

  describe "#slice" do
    it "slices a region of the array" do
      c = MXNet::Symbol.var("c")
      c.slice(begin: [0, 0, 1], end: [1, 2, 2]).eval(**args).first.should eq(MXNet::NDArray.array([[[2.0], [4.0]]]))
    end
  end

  describe "#slice_axis" do
    it "slices a region of the array" do
      c = MXNet::Symbol.var("c")
      c.slice_axis(axis: 2, begin: 0, end: 1).eval(**args).first.should eq(MXNet::NDArray.array([[[1.0], [3.0], [5.0], [7.0]]]))
    end
  end

  describe "#softmax" do
    it "applies the softmax function" do
      a = MXNet::Symbol.var("a")
      a.softmax(axis: 0).eval(**args).first.should be_close(MXNet::NDArray.array([[0.1192, 0.1192], [0.8807, 0.8807]]), 0.05)
      a.softmax(axis: 1).eval(**args).first.should be_close(MXNet::NDArray.array([[0.2689, 0.7310], [0.2689, 0.7310]]), 0.05)
    end
  end

  describe "#sqrt" do
    it "computes the square-root of the input" do
      d = MXNet::Symbol.var("d")
      d.sqrt.eval(**args).first.should eq(MXNet::NDArray.array([[1.0], [2.0], [3.0]]))
    end
  end

  describe "#square" do
    it "computes the square of the input" do
      a = MXNet::Symbol.var("a")
      a.square.eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 4.0], [9.0, 16.0]]))
    end
  end

  describe "#sum" do
    it "computes the sum" do
      c = MXNet::Symbol.var("c")
      c.sum(axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[16.0, 20.0]]))
      c.sum(axis: 2).eval(**args).first.should eq(MXNet::NDArray.array([[3.0, 7.0, 11.0, 15.0]]))
      c.sum.eval(**args).first.should eq(MXNet::NDArray.array([36.0]))
    end
  end

  describe "#take" do
    it "takes elements from an input array" do
      a = MXNet::Symbol.var("a")
      e = MXNet::Symbol.var("e")
      a.take(e, axis: 1).eval(**args).first.should eq(MXNet::NDArray.array([[[1.0], [2.0]], [[3.0], [4.0]]]))
    end
  end

  describe "#transpose" do
    it "permutes the dimensions of the array" do
      a = MXNet::Symbol.var("a")
      a.transpose.eval(**args).first.should eq(MXNet::NDArray.array([[1.0, 3.0], [2.0, 4.0]]))
      c = MXNet::Symbol.var("c")
      c.transpose.eval(**args).first.should eq(MXNet::NDArray.array([[[1.0], [3.0], [5.0], [7.0]], [[2.0], [4.0], [6.0], [8.0]]]))
      c.transpose(axes: [1, 0, 2]).eval(**args).first.should eq(MXNet::NDArray.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]))
    end
  end

  describe ".zeros_like" do
    it "creates an array of the same shape filled with zeros" do
      a = MXNet::Symbol.var("a")
      z = MXNet::Symbol.zeros_like(a)
      z.eval(**args).first.should eq(MXNet::NDArray.array([[0.0, 0.0], [0.0, 0.0]]))
    end
  end
end
