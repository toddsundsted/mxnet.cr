require "../spec_helper"

private macro random_spec_helper(random, *args, dtype = :float32)
  describe ".{{random}}" do
    it "returns an array of random numbers" do
      MXNet::NDArray.{{random}}({{*args}}, shape: 2).shape.should eq([2])
      MXNet::NDArray.{{random}}({{*args}}, shape: [1, 2, 3]).shape.should eq([1, 2, 3])
      MXNet::NDArray.{{random}}({{*args}}, shape: 1, dtype: {{dtype}}).dtype.should eq({{dtype}})
    end

    if gpu_enabled?
      it "uses the specified context" do
        MXNet::NDArray.{{random}}({{*args}}, shape: 1, ctx: MXNet.gpu(0)).context.should eq(MXNet.gpu(0))
      end

      it "uses the current context" do
        MXNet::Context.with(MXNet.gpu(0)) do
          MXNet::NDArray.{{random}}({{*args}}, shape: 1).context.should eq(MXNet.gpu(0))
        end
      end
    end

    it "writes the results to the output array" do
      {% dtype = dtype.stringify[1..-1].tr("3264", "6432") %}
      a = MXNet::NDArray.empty(1, dtype: {{dtype.id.symbolize}})
      b = MXNet::NDArray.{{random}}({{*args}}, out: a)
      a.as_scalar.should be_a({{dtype.capitalize.id}})
      a.should be(b)
    end
  end
end

private macro sample_spec_helper(sample, *args, return_types = {Float64, :float64})
  describe ".{{sample}}" do
    args = {
      {% for arg in (args) %}
        MXNet::NDArray.array({{arg}}),
      {% end %}
    }

    it "returns an array of sampled numbers" do
      MXNet::NDArray.{{sample}}(*args, shape: 2).shape.should eq([2, 2])
      MXNet::NDArray.{{sample}}(*args, shape: [1, 2, 3]).shape.should eq([2, 1, 2, 3])
      MXNet::NDArray.{{sample}}(*args, shape: 1, dtype: {{return_types[1]}}).dtype.should eq({{return_types[1]}})
    end

    if gpu_enabled?
      it "infers context from arguments" do
        MXNet::NDArray.{{sample}}(*args.map(&.as_in_context(MXNet.gpu(0))), shape: 1).context.should eq(MXNet.gpu(0))
      end
    end

    it "writes the results to the output array" do
      a = MXNet::NDArray.empty(2, dtype: {{return_types[1]}})
      b = MXNet::NDArray.{{sample}}(*args, out: a)
      a.first.as_scalar.should be_a({{return_types[0]}})
      a.should be(b)
    end
  end
end

describe MXNet::NDArray do
  describe ".imperative_invoke" do
    it "removes nil arguments" do
      a = MXNet::NDArray.array([1.0, 2.0, 3.0])
      MXNet::NDArray.imperative_invoke("Dropout", a, p: nil)
    end

    it "removes nil arguments" do
      a = MXNet::NDArray.array([1, 2, 3])
      MXNet::NDArray.imperative_invoke("elemwise_add", a, a, nil)
    end

    it "flattens arguments" do
      a = MXNet::NDArray.array([1, 2, 3])
      MXNet::NDArray.imperative_invoke("add_n", [a, a], num_args: 2)
    end
  end

  describe ".save" do
    temp = File.tempname("save")
    array = MXNet::NDArray.array([1, 2, 3], dtype: :float32)

    it "saves an array" do
      data = Bytes[
        0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x40, 0x40, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x61
      ]
      MXNet::NDArray.save(temp, {"a" => array})
      buffer = Bytes.new(85)
      File.open(temp, "r") do |file|
        file.read(buffer)
        buffer.should eq(data)
      end
    end

    it "saves an array" do
      data = Bytes[
        0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
      ]
      MXNet::NDArray.save(temp, [array])
      buffer = Bytes.new(76)
      File.open(temp, "r") do |file|
        file.read(buffer)
        buffer.should eq(data)
      end
    end
  end

  describe ".load" do
    temp = File.tempname("load")
    array = MXNet::NDArray.array([1, 2, 3], dtype: :float32)

    it "loads an array" do
      data = Bytes[
        0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x40, 0x40, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x61
      ]
      File.open(temp, "w") do |file|
        file.write(data)
        file.close
        MXNet::NDArray.load(temp).should eq({"a" => array})
      end
    end

    it "loads an array" do
      data = Bytes[
        0x12, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xc9, 0xfa, 0x93, 0xf9, 0x00, 0x00, 0x00, 0x00,
        0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
      ]
      File.open(temp, "w") do |file|
        file.write(data)
        file.close
        MXNet::NDArray.load(temp).should eq([array])
      end
    end
  end

  describe ".array" do
    it "creates an NDArray from a Crystal array" do
      MXNet::NDArray.array([[[[1]]]]).should be_a(MXNet::NDArray)
      MXNet::NDArray.array([[[1]]]).should be_a(MXNet::NDArray)
      MXNet::NDArray.array([[1]]).should be_a(MXNet::NDArray)
      MXNet::NDArray.array([1]).should be_a(MXNet::NDArray)
    end

    it "supports MXNet numeric types" do
      MXNet::NDArray.array([1], dtype: :float32).should eq(MXNet::NDArray.array([1_f32]))
      MXNet::NDArray.array([1], dtype: :float64).should eq(MXNet::NDArray.array([1_f64]))
      MXNet::NDArray.array([1], dtype: :uint8).should eq(MXNet::NDArray.array([1_u8]))
      MXNet::NDArray.array([1], dtype: :int32).should eq(MXNet::NDArray.array([1_i32]))
      MXNet::NDArray.array([1], dtype: :int8).should eq(MXNet::NDArray.array([1_i8]))
      MXNet::NDArray.array([1], dtype: :int64).should eq(MXNet::NDArray.array([1_i64]))
    end

    it "supports Crystal numeric types" do
      MXNet::NDArray.array([1_f32]).should eq(MXNet::NDArray.array([1_f32]))
      MXNet::NDArray.array([1_f64]).should eq(MXNet::NDArray.array([1_f64]))
      MXNet::NDArray.array([1_u8]).should eq(MXNet::NDArray.array([1_u8]))
      MXNet::NDArray.array([1_i32]).should eq(MXNet::NDArray.array([1_i32]))
      MXNet::NDArray.array([1_i8]).should eq(MXNet::NDArray.array([1_i8]))
      MXNet::NDArray.array([1_i64]).should eq(MXNet::NDArray.array([1_i64]))
    end

    it "supports explicit context" do
      MXNet::NDArray.array([1], ctx: MXNet::Context.cpu).context.should eq(MXNet::Context.cpu)
    end

    it "fails if the array has inconsistent nesting" do
      expect_raises(MXNet::NDArrayException, /inconsistent nesting/) do
        MXNet::NDArray.array([1, [2.0]])
      end
    end

    it "fails if slices along an axis have different dimensions" do
      expect_raises(MXNet::NDArrayException, /inconsistent dimensions/) do
        MXNet::NDArray.array([[1, 2], [3]])
      end
    end

    it "fails if the array isn't an array of numeric types" do
      expect_raises(MXNet::NDArrayException, /type is unsupported.*String/) do
        MXNet::NDArray.array(["one", "two"])
      end
    end
  end

  describe ".zeros" do
    it "returns a new array filled with all zeros" do
      MXNet::NDArray.zeros([2, 3]).should eq(MXNet::NDArray.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype: :float32))
      MXNet::NDArray.zeros(2, dtype: :int32).should eq(MXNet::NDArray.array([0, 0], dtype: :int32))
    end

    if gpu_enabled?
      it "uses the specified context" do
        MXNet::NDArray.zeros(2, ctx: MXNet.gpu(0)).context.should eq(MXNet.gpu(0))
       end

      it "uses the current context" do
        MXNet::Context.with(MXNet.gpu(0)) do
          MXNet::NDArray.zeros(1).context.should eq(MXNet.gpu(0))
        end
      end
    end

    it "writes the results to the output array" do
      a = MXNet::NDArray.array([99_f32])
      b = MXNet::NDArray.zeros(1, out: a)
      a.should eq(MXNet::NDArray.array([0_f32]))
      a.should be(b)
    end
  end

  describe ".ones" do
    it "returns a new array filled with all ones" do
      MXNet::NDArray.ones([2, 3]).should eq(MXNet::NDArray.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype: :float32))
      MXNet::NDArray.ones(2, dtype: :int64).should eq(MXNet::NDArray.array([1, 1], dtype: :int64))
    end

    if gpu_enabled?
      it "uses the specified context" do
        MXNet::NDArray.ones(2, ctx: MXNet.gpu(0)).context.should eq(MXNet.gpu(0))
       end

      it "uses the current context" do
        MXNet::Context.with(MXNet.gpu(0)) do
          MXNet::NDArray.ones(1).context.should eq(MXNet.gpu(0))
        end
      end
    end

    it "writes the results to the output array" do
      a = MXNet::NDArray.array([99_f32])
      b = MXNet::NDArray.ones(1, out: a)
      a.should eq(MXNet::NDArray.array([1_f32]))
      a.should be(b)
    end
  end

  random_spec_helper(random_uniform, 0.0, 1.0)
  random_spec_helper(random_normal, 0.0, 1.0)
  random_spec_helper(random_poisson, 1.0)
  random_spec_helper(random_exponential, 1.0)
  random_spec_helper(random_gamma, 1.0, 1.0)
  {% unless compare_versions(MXNet::Internal::MXNET_VERSION, "1.4.0") < 0 %}
    random_spec_helper(
      random_randint, 1, 9,
      dtype: :int32
    )
  {% end %}

  sample_spec_helper(sample_uniform, [0.0, 2.5], [1.0, 3.7])
  sample_spec_helper(sample_normal, [0.0, 2.5], [1.0, 3.7])
  sample_spec_helper(sample_poisson, [1.0, 8.5])
  sample_spec_helper(sample_exponential, [1.0, 8.5])
  sample_spec_helper(sample_gamma, [0.0, 2.5], [1.0, 0.7])
  sample_spec_helper(
    sample_multinomial,
    [[0.0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.0]],
    return_types: {Int32, :int32}
  )

  describe "#shape" do
    it "returns the shape of the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).shape.should eq([3])
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).shape.should eq([2, 2])
      MXNet::NDArray.array([1, 2]).shape.should eq([2])
      MXNet::NDArray.array([1_u8]).shape.should eq([1])
    end
  end

  describe "#context" do
    it "returns the context of the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).context.should eq(MXNet::Context.cpu)
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).context.should eq(MXNet::Context.cpu)
      MXNet::NDArray.array([1, 2]).context.should eq(MXNet::Context.cpu)
      MXNet::NDArray.array([1_u8]).context.should eq(MXNet::Context.cpu)
    end
  end

  describe "#dtype" do
    it "returns the dtype of the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).dtype.should eq(:float64)
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).dtype.should eq(:int64)
      MXNet::NDArray.array([1, 2]).dtype.should eq(:int32)
      MXNet::NDArray.array([1_u8]).dtype.should eq(:uint8)
    end
  end

  describe "#copy_to" do
    it "fails if the source and destination are the same" do
      expect_raises(MXNet::NDArrayException, "cannot copy an array onto itself") do
        a = MXNet::NDArray.array([1, 2, 3])
        a.copy_to(a)
      end
    end
  end

  describe "#as_type" do
    it "returns a copy after casting to the specified type" do
      MXNet::NDArray.array([1]).as_type(:float32).should eq(MXNet::NDArray.array([1_f32]))
      MXNet::NDArray.array([1]).as_type(:float64).should eq(MXNet::NDArray.array([1_f64]))
      MXNet::NDArray.array([1]).as_type(:uint8).should eq(MXNet::NDArray.array([1_u8]))
      MXNet::NDArray.array([1]).as_type(:int32).should eq(MXNet::NDArray.array([1_i32]))
      MXNet::NDArray.array([1]).as_type(:int8).should eq(MXNet::NDArray.array([1_i8]))
      MXNet::NDArray.array([1]).as_type(:int64).should eq(MXNet::NDArray.array([1_i64]))
    end
  end

  describe "#as_in_context" do
    if gpu_enabled?
      it "returns a copy on the target device" do
        a = MXNet::NDArray.array([1]).as_in_context(MXNet.gpu(0))
        a.should eq(MXNet::NDArray.array([1_i32]))
        a.context.should eq(MXNet.gpu(0))
      end
    end
  end

  describe "#as_scalar" do
    it "returns a scalar" do
      MXNet::NDArray.zeros([1], dtype: :float64).as_scalar.should eq(0.0)
    end

    it "fails if shape != [1]" do
      expect_raises(MXNet::NDArrayException, "the array is not scalar") do
        MXNet::NDArray.zeros([4]).as_scalar
      end
    end
  end

  describe "#to_a" do
    it "returns an array" do
      MXNet::NDArray.zeros([4], dtype: :float64).to_a.should eq([0.0, 0.0, 0.0, 0.0])
    end

    it "fails if shape.size > 1" do
      expect_raises(MXNet::NDArrayException, "the array must have only 1 dimension") do
        MXNet::NDArray.zeros([2, 2]).to_a
      end
    end
  end

  describe "#to_s" do
    it "pretty-prints the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).to_s.should eq("[1.0, 2.0, 3.0]\n<NDArray 3 float64 cpu(0)>")
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).to_s.should eq("[[1, 2], [3, 4]]\n<NDArray 2x2 int64 cpu(0)>")
      MXNet::NDArray.array([1, 2]).to_s.should eq("[1, 2]\n<NDArray 2 int32 cpu(0)>")
      MXNet::NDArray.array([1_u8]).to_s.should eq("[1]\n<NDArray 1 uint8 cpu(0)>")
    end
  end

  describe "#grad" do
    it "returns the attached gradient buffer" do
      MXNet::NDArray.array([1]).attach_grad.grad.should eq(MXNet::NDArray.array([0]))
    end

    it "fails if no gradient buffer is attached" do
      expect_raises(MXNet::NDArrayException, "no gradient is attached") do
        MXNet::NDArray.array([1]).grad
      end
    end
  end

  describe "#attach_grad" do
    it "attaches a gradient buffer" do
      MXNet::NDArray.array([1]).attach_grad.grad.should be_a(MXNet::NDArray)
    end
  end

  describe "#+" do
    it "adds a scalar to an array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a + 2).should eq(MXNet::NDArray.array([[3.0, 4.0], [5.0, 6.0]]))
      (2 + a).should eq(MXNet::NDArray.array([[3.0, 4.0], [5.0, 6.0]]))
    end
    it "adds two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a + b).should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
    context "unary" do
      it "leaves the array unchanged" do
        (+MXNet::NDArray.array([1.0, 2.0])).should eq(MXNet::NDArray.array([1.0, 2.0]))
        (+MXNet::NDArray.array([1, 2])).should eq(MXNet::NDArray.array([1, 2]))
      end
    end
  end

  describe "#-" do
    it "subtracts a scalar from an array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a - 2).should eq(MXNet::NDArray.array([[-1.0, 0.0], [1.0, 2.0]]))
      (2 - a).should eq(MXNet::NDArray.array([[1.0, 0.0], [-1.0, -2.0]]))
    end
    it "subtracts two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a - b).should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
    context "unary" do
      it "negates the array" do
        (-MXNet::NDArray.array([1.0, 2.0])).should eq(MXNet::NDArray.array([-1.0, -2.0]))
        (-MXNet::NDArray.array([1, 2])).should eq(MXNet::NDArray.array([-1, -2]))
      end
    end
  end

  describe "#*" do
    it "multiplies an array by a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a * 2).should eq(MXNet::NDArray.array([[2.0, 4.0], [6.0, 8.0]]))
      (2 * a).should eq(MXNet::NDArray.array([[2.0, 4.0], [6.0, 8.0]]))
    end
    it "multiplies two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a * b).should eq(MXNet::NDArray.array([[1.0, 8.0], [3.0, 4.0]]))
    end
  end

  describe "#/" do
    it "divides an array by a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a / 2).should eq(MXNet::NDArray.array([[0.5, 1.0], [1.5, 2.0]]))
      (2 / a).should eq(MXNet::NDArray.array([[2.0, 1.0], [(1.0/1.5), 0.5]]))
    end
    it "divides two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a / b).should eq(MXNet::NDArray.array([[1.0, 0.5], [3.0, 4.0]]))
    end
  end

  describe "#**" do
    it "exponentiates an array by a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a ** 2).should eq(MXNet::NDArray.array([[1.0, 4.0], [9.0, 16.0]]))
      (2 ** a).should eq(MXNet::NDArray.array([[2.0, 4.0], [8.0, 16.0]]))
    end
    it "exponentiates two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a ** b).should eq(MXNet::NDArray.array([[1.0, 16.0], [3.0, 4.0]]))
    end
  end

  describe "#==" do
    it "performs element-wise equal with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a == 2).should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
      (2 == a).should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
    it "performs element-wise equal" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a == b).should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
  end

  describe "#!=" do
    it "performs element-wise not equal with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a != 2).should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
      (2 != a).should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
    it "performs element-wise not equal" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a != b).should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe "#>" do
    it "performs element-wise greater than with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a > 2).should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
      (2 > a).should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
    it "performs element-wise greater than" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a > b).should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe "#>=" do
    it "performs element-wise greater than or equal to with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a >= 2).should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
      (2 >= a).should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
    it "performs element-wise greater than or equal to" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a >= b).should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe "#<" do
    it "performs element-wise less than with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a < 2).should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
      (2 < a).should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
    it "performs element-wise less than" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a < b).should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe "#<=" do
    it "performs element-wise less than or equal to with a scalar" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      (a <= 2).should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
      (2 <= a).should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
    it "performs element-wise less than or equal to" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      (a <= b).should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".abs" do
    it "computes the element-wise absolute value of the input" do
      e = MXNet::NDArray.array([[-1.0], [1.0]])
      e.abs.should eq(MXNet::NDArray.array([[1.0], [1.0]]))
    end
  end

  describe ".add_n" do
    it "adds arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      MXNet::NDArray.add_n([a, b]).should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".arange" do
    it "returns evenly spaced values within a given interval" do
      MXNet::NDArray.arange(3).should eq(MXNet::NDArray.array([0.0, 1.0, 2.0], :float32))
      MXNet::NDArray.arange(2, 6).should eq(MXNet::NDArray.array([2.0, 3.0, 4.0, 5.0], :float32))
      MXNet::NDArray.arange(2, 6, step: 2).should eq(MXNet::NDArray.array([2.0, 4.0], :float32))
    end
  end

  describe ".broadcast_add" do
    it "adds two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_add(b).should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".broadcast_div" do
    it "divides two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_div(b).should eq(MXNet::NDArray.array([[1.0, 0.5], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_equal" do
    it "performs element-wise equal" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_equal(b).should eq(MXNet::NDArray.array([[1.0, 0.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_greater" do
    it "performs element-wise greater than" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_greater(b).should eq(MXNet::NDArray.array([[0.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_greater_equal" do
    it "performs element-wise greater than or equal to" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_greater_equal(b).should eq(MXNet::NDArray.array([[1.0, 0.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_lesser" do
    it "performs element-wise less than" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_lesser(b).should eq(MXNet::NDArray.array([[0.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_lesser_equal" do
    it "performs element-wise less than or equal to" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_lesser_equal(b).should eq(MXNet::NDArray.array([[1.0, 1.0], [0.0, 0.0]]))
    end
  end

  describe ".broadcast_maximum" do
    it "returns the maximum" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_maximum(b).should eq(MXNet::NDArray.array([[1.0, 4.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_minimum" do
    it "returns the minimum" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_minimum(b).should eq(MXNet::NDArray.array([[1.0, 2.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_minus" do
    it "subtracts two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_minus(b).should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
  end

  describe ".broadcast_mul" do
    it "multiplies two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_mul(b).should eq(MXNet::NDArray.array([[1.0, 8.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_not_equal" do
    it "performs element-wise not equal" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_not_equal(b).should eq(MXNet::NDArray.array([[0.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe ".broadcast_plus" do
    it "adds two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_plus(b).should eq(MXNet::NDArray.array([[2.0, 6.0], [4.0, 5.0]]))
    end
  end

  describe ".broadcast_power" do
    it "exponentiates two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_power(b).should eq(MXNet::NDArray.array([[1.0, 16.0], [3.0, 4.0]]))
    end
  end

  describe ".broadcast_sub" do
    it "subtracts two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      a.broadcast_sub(b).should eq(MXNet::NDArray.array([[0.0, -2.0], [2.0, 3.0]]))
    end
  end

  describe ".clip" do
    it "clips the values in an array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.clip(2.0, 7.0).should eq(MXNet::NDArray.array([[[2.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 7.0]]]))
    end
  end

  describe ".concat" do
    it "concatenates arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      MXNet::NDArray.concat([a, b]).should eq(MXNet::NDArray.array([[1.0, 2.0, 1.0, 4.0], [3.0, 4.0, 1.0, 1.0]]))
    end
  end

  describe ".dot" do
    it "computes the dot product of two arrays" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      x = a.reshape(shape: [2, 2])
      y = a.reshape(shape: 4).flip(axis: 0).reshape(shape: [2, 2])
      MXNet::NDArray.dot(x, y).should eq(MXNet::NDArray.array([[8.0, 5.0], [20.0, 13.0]]))
    end
  end

  describe "#exp" do
    it "computes the exponential" do
      i = MXNet::NDArray.array([0.0, 1.0])
      i.exp.should be_close(MXNet::NDArray.array([1.0000, 2.7182]), 0.001)
    end
  end

  describe "#expand_dims" do
    it "inserts a new axis into the input array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.expand_dims(axis: 1).shape.should eq([1, 1, 4, 2])
      c.expand_dims(1).shape.should eq([1, 1, 4, 2])
    end
  end

  describe "#flatten" do
    it "flattens the input array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.flatten.shape.should eq([1, 8])
    end
  end

  describe "#flip" do
    it "reverses the order of elements" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.flip(axis: 1).should eq(MXNet::NDArray.array([[[7.0, 8.0], [5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]]))
      c.flip(axis: 2).should eq(MXNet::NDArray.array([[[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]]))
    end
  end

  describe "#floor" do
    it "returns floor of the input" do
      f = MXNet::NDArray.array([-2.1, -1.9, 1.5, 1.9, 2.1])
      f.floor.should eq(MXNet::NDArray.array([-3.0, -2.0, 1.0, 1.0, 2.0]))
    end
  end

  describe "#log" do
    it "computes the natural logarithm" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.log.should be_close(MXNet::NDArray.array([[0.0, 0.69314], [1.0986, 1.3862]]), 0.001)
    end
  end

  describe "#log_softmax" do
    it "computes the log softmax of the input" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.log_softmax(axis: 0).should be_close(MXNet::NDArray.array([[-2.1269, -2.1269], [-0.1269, -0.1269]]), 0.05)
      a.log_softmax(axis: 1).should be_close(MXNet::NDArray.array([[-1.3133, -0.3133], [-1.3133, -0.3133]]), 0.05)
    end
  end

  describe "#max" do
    it "computes the max" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.max(axis: 1).should eq(MXNet::NDArray.array([[7.0, 8.0]]))
      c.max(axis: 2).should eq(MXNet::NDArray.array([[2.0, 4.0, 6.0, 8.0]]))
      c.max.should eq(MXNet::NDArray.array([8.0]))
    end
  end

  describe "#mean" do
    it "computes the mean" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.mean(axis: 1).should eq(MXNet::NDArray.array([[4.0, 5.0]]))
      c.mean(axis: 2).should eq(MXNet::NDArray.array([[1.5, 3.5, 5.5, 7.5]]))
      c.mean.should eq(MXNet::NDArray.array([4.5]))
    end
  end

  describe "#min" do
    it "computes the min" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.min(axis: 1).should eq(MXNet::NDArray.array([[1.0, 2.0]]))
      c.min(axis: 2).should eq(MXNet::NDArray.array([[1.0, 3.0, 5.0, 7.0]]))
      c.min.should eq(MXNet::NDArray.array([1.0]))
    end
  end

  describe "#norm" do
    it "returns the norm" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.norm.should be_close(MXNet::NDArray.array([5.47]), 0.05)
    end
  end

  describe "#one_hot" do
    it "returns a one-hot array" do
      b = MXNet::NDArray.array([[1.0, 4.0], [1.0, 1.0]])
      o = MXNet::NDArray.array([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype: :float32)
      b.reshape(shape: [-1]).one_hot(5).should eq(o)
    end
  end

  describe ".ones_like" do
    it "creates an array of the same shape filled with ones" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      z = MXNet::NDArray.ones_like(a)
      z.should eq(MXNet::NDArray.array([[1.0, 1.0], [1.0, 1.0]]))
    end
  end

  describe "#pick" do
    it "picks elements from an input array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      i = MXNet::NDArray.array([0.0, 1.0])
      a.pick(i, axis: 0).should eq(MXNet::NDArray.array([1.0, 4.0]))
    end
  end

  describe "#relu" do
    it "computes the rectified linear activation of the input" do
      e = MXNet::NDArray.array([[-1.0], [1.0]])
      e.relu.should eq(MXNet::NDArray.array([[0.0], [1.0]]))
    end
  end

  describe "#reshape" do
    it "reshapes the input array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.reshape(shape: [4]).should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape(shape: 4).should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
    end

    it "supports special values for dimensions" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.reshape(shape: [-1, 0], reverse: false).shape.should eq([2, 4])
      c.reshape(shape: [-1, 0], reverse: true).shape.should eq([4, 2])
    end
  end

  describe "#reshape_like" do
    it "reshapes the input array" do
      a = MXNet::NDArray.zeros(shape: [9])
      b = MXNet::NDArray.zeros(shape: [3, 3])
      a.reshape_like(b).shape.should eq([3, 3])
    end
  end

  describe ".shuffle" do
    it "randomly shuffles the elements" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      MXNet::NDArray.shuffle(a).should be_a(MXNet::NDArray)
    end
  end

  describe "#sigmoid" do
    it "computes the sigmoid activation of the input" do
      z = MXNet::NDArray.zeros([1])
      z.sigmoid.as_scalar.should eq(0.5)
    end
  end

  describe "#sign" do
    it "returns element-wise sign of the input" do
      e = MXNet::NDArray.array([[-1.0], [1.0]])
      e.sign.should eq(MXNet::NDArray.array([[-1.0], [1.0]]))
    end
  end

  describe "#slice" do
    it "slices a region of the array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.slice(begin: [0, 0, 1], end: [1, 2, 2]).should eq(MXNet::NDArray.array([[[2.0], [4.0]]]))
    end
  end

  describe "#slice_axis" do
    it "slices a region of the array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.slice_axis(axis: 2, begin: 0, end: 1).should eq(MXNet::NDArray.array([[[1.0], [3.0], [5.0], [7.0]]]))
    end
  end

  describe "#softmax" do
    it "applies the softmax function" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.softmax(axis: 0).should be_close(MXNet::NDArray.array([[0.1192, 0.1192], [0.8807, 0.8807]]), 0.05)
      a.softmax(axis: 1).should be_close(MXNet::NDArray.array([[0.2689, 0.7310], [0.2689, 0.7310]]), 0.05)
    end
  end

  describe "#sqrt" do
    it "computes the square-root of the input" do
      d = MXNet::NDArray.array([[1.0], [4.0], [9.0]])
      d.sqrt.should eq(MXNet::NDArray.array([[1.0], [2.0], [3.0]]))
    end
  end

  describe "#square" do
    it "computes the square of the input" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.square.should eq(MXNet::NDArray.array([[1.0, 4.0], [9.0, 16.0]]))
    end
  end

  describe "#sum" do
    it "computes the sum" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.sum(axis: 1).should eq(MXNet::NDArray.array([[16.0, 20.0]]))
      c.sum(axis: 2).should eq(MXNet::NDArray.array([[3.0, 7.0, 11.0, 15.0]]))
      c.sum.should eq(MXNet::NDArray.array([36.0]))
    end
  end

  describe "#take" do
    it "takes elements from an input array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      e = MXNet::NDArray.array([[-1.0], [1.0]])
      a.take(e).should eq(MXNet::NDArray.array([[[1.0, 2.0]], [[3.0, 4.0]]]))
    end
  end

  describe "#transpose" do
    it "permutes the dimensions of the array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.transpose.should eq(MXNet::NDArray.array([[1.0, 3.0], [2.0, 4.0]]))
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.transpose.should eq(MXNet::NDArray.array([[[1.0], [3.0], [5.0], [7.0]], [[2.0], [4.0], [6.0], [8.0]]]))
      c.transpose(axes: [1, 0, 2]).should eq(MXNet::NDArray.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]))
    end
  end

  describe ".zeros_like" do
    it "creates an array of the same shape filled with zeros" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      z = MXNet::NDArray.zeros_like(a)
      z.should eq(MXNet::NDArray.array([[0.0, 0.0], [0.0, 0.0]]))
    end
  end

  describe "#[]" do
    it "returns the specified slice along the first axis" do
      a = MXNet::NDArray.array([1, 2, 3, 4])
      a[1].should eq(MXNet::NDArray.array([2]))
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1].should eq(MXNet::NDArray.array([[5, 6], [7, 8]]))
    end

    it "returns the specified slices along the first axis" do
      a = MXNet::NDArray.array([1, 2, 3, 4])
      a[1...3].should eq(MXNet::NDArray.array([2, 3]))
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1...3].should eq(MXNet::NDArray.array([[[5, 6], [7, 8]], [[9, 0], [1, 2]]]))
    end

    it "returns the specified slices along the first axis" do
      a = MXNet::NDArray.array([1, 2, 3, 4])
      a[1..-2].should eq(MXNet::NDArray.array([2, 3]))
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1..-1].should eq(MXNet::NDArray.array([[[5, 6], [7, 8]], [[9, 0], [1, 2]]]))
    end

    it "returns the specified value" do
      a = MXNet::NDArray.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]])
      a[2, 3].should eq(MXNet::NDArray.array([11]))
    end

    it "supports mixed ranges and indexes" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1...3, 1].should eq(MXNet::NDArray.array([[7, 8], [1, 2]]))
    end

    it "supports mixed ranges and indexes" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1...3, 1...2].should eq(MXNet::NDArray.array([[[7, 8]], [[1, 2]]]))
    end

    it "supports mixed ranges and indexes" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1, 1...2].should eq(MXNet::NDArray.array([[7, 8]]))
    end

    it "supports mixed ranges and indexes" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1, 1].should eq(MXNet::NDArray.array([7, 8]))
    end

    it "supports open ranges" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[1, ..].should eq(MXNet::NDArray.array([[5, 6], [7, 8]]))
    end

    it "supports open ranges" do
      b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
      b[.., 1].should eq(MXNet::NDArray.array([[3, 4], [7, 8], [1, 2]]))
    end

    it "reduces dimensionality correctly" do
      x = MXNet::NDArray.array((0...7 * 5 * 3 * 2).to_a).reshape(shape: [7, 5, 3, 2])
      x[1].shape.should eq([5, 3, 2])
      x[1, 1].shape.should eq([3, 2])
      x[1, 1, 1].shape.should eq([2])
      x[1, 1, 1, 1].shape.should eq([1])
      x[0..-1, 1].shape.should eq([7, 3, 2])
      x[0..-1, 0..-1, 1].shape.should eq([7, 5, 2])
      x[0..-1, 1, 0..-1].shape.should eq([7, 3, 2])
      x[0..-1, 1, 1].shape.should eq([7, 2])
    end
  end

  describe "#[]=" do
    context "value is an array" do
      it "replaces the specified slice along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1] = MXNet::NDArray.array([99])
        a.should eq(MXNet::NDArray.array([1, 99, 3, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1] = MXNet::NDArray.array([[99, 99], [99, 99]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "replaces the specified slices along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1...3] = MXNet::NDArray.array([99, 99])
        a.should eq(MXNet::NDArray.array([1, 99, 99, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3] = MXNet::NDArray.array([[[99, 99], [99, 99]], [[99, 99], [99, 99]]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[99, 99], [99, 99]]]))
      end

      it "replaces the specified slices along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1..-2] = MXNet::NDArray.array([99, 99])
        a.should eq(MXNet::NDArray.array([1, 99, 99, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1..-1] = MXNet::NDArray.array([[[99, 99], [99, 99]], [[99, 99], [99, 99]]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[99, 99], [99, 99]]]))
      end

      it "replaces the specified value" do
        a = MXNet::NDArray.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]])
        a[2, 3] = MXNet::NDArray.array([99])
        a.should eq(MXNet::NDArray.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 99], [12, 13, 14, 15], [16, 17, 18, 19]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3, 1] = MXNet::NDArray.array([[99, 99], [99, 99]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3, 1...2] = MXNet::NDArray.array([[[99, 99]], [[99, 99]]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1, 1...2] = MXNet::NDArray.array([[99, 99]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1, 1] = MXNet::NDArray.array([99, 99])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "supports open ranges" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1, ..] = MXNet::NDArray.array([[99, 99], [99, 99]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "supports open ranges" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[.., 1] = MXNet::NDArray.array([[99, 99], [99, 99], [99, 99]])
        b.should eq(MXNet::NDArray.array([[[1, 2], [99, 99]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]]))
      end
    end

    context "value is a scalar" do
      it "replaces the specified slice along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1] = 99
        a.should eq(MXNet::NDArray.array([1, 99, 3, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "replaces the specified slices along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1...3] = 99
        a.should eq(MXNet::NDArray.array([1, 99, 99, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[99, 99], [99, 99]]]))
      end

      it "replaces the specified slices along the first axis" do
        a = MXNet::NDArray.array([1, 2, 3, 4])
        a[1..-2] = 99
        a.should eq(MXNet::NDArray.array([1, 99, 99, 4]))
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1..-1] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[99, 99], [99, 99]], [[99, 99], [99, 99]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3, 1] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1...3, 1...2] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [99, 99]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1, 1...2] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [1, 2]]]))
      end

      it "supports mixed ranges and indexes" do
        b = MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 0], [1, 2]]])
        b[1, 1] = 99
        b.should eq(MXNet::NDArray.array([[[1, 2], [3, 4]], [[5, 6], [99, 99]], [[9, 0], [1, 2]]]))
      end
    end
  end
end
