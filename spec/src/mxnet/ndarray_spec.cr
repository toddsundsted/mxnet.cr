require "../../spec_helper"

class MXNet::NDArray
  # Redefined this method for testing. The original method
  # performs an element-wise comparison for equality.
  def ==(other : self)
    return false unless other.dtype == self.dtype
    return false unless other.shape == self.shape
    return false unless other.raw == self.raw
    true
  end
end

describe "MXNet::NDArray" do
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

    it "fails if the array isn't an array of a supported type" do
      expect_raises(MXNet::NDArrayException, /type is unsupported.*String/) do
        MXNet::NDArray.array(["one", "two"])
      end
    end

    it "fails if the array type is a union type" do
      expect_raises(MXNet::NDArrayException, /type is unsupported/) do
        MXNet::NDArray.array([[1.0], [2]])
      end
      expect_raises(MXNet::NDArrayException, /type is unsupported/) do
        MXNet::NDArray.array([[1], [2.0]])
      end
      expect_raises(MXNet::NDArrayException, /type is unsupported/) do
        MXNet::NDArray.array([1, 2.0])
      end
    end
  end

  describe ".zeros" do
    it "returns a new array filled with all zeros" do
      MXNet::NDArray.zeros([2, 3]).should eq(MXNet::NDArray.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype: :float32))
      MXNet::NDArray.zeros(2, dtype: :int32).should eq(MXNet::NDArray.array([0, 0], dtype: :int32))
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

    it "writes the results to the output array" do
      a = MXNet::NDArray.array([99_f32])
      b = MXNet::NDArray.ones(1, out: a)
      a.should eq(MXNet::NDArray.array([1_f32]))
      a.should be(b)
    end
  end

  describe ".random_uniform" do
    it "returns an array of random numbers" do
      MXNet::NDArray.random_uniform(0.0, 1.0, shape: [1, 2, 3], dtype: :float32, ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
  end

  describe ".random_normal" do
    it "returns an array of random numbers" do
      MXNet::NDArray.random_normal(0.0, 1.0, shape: [1, 2, 3], dtype: :float32, ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
  end

  describe ".random_poisson" do
    it "returns an array of random numbers" do
      MXNet::NDArray.random_poisson(1.0, shape: [1, 2, 3], dtype: :float32, ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
  end

  describe ".random_exponential" do
    it "returns an array of random numbers" do
      MXNet::NDArray.random_exponential(1.0, shape: [1, 2, 3], dtype: :float32, ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
  end

  describe ".random_gamma" do
    it "returns an array of random numbers" do
      MXNet::NDArray.random_gamma(1.0, 1.0, shape: [1, 2, 3], dtype: :float32, ctx: MXNet.cpu).should be_a(MXNet::NDArray)
    end
  end

  describe "#shape" do
    it "returns the shape of the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).shape.should eq([3_u32])
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).shape.should eq([2_u32, 2_u32])
      MXNet::NDArray.array([1, 2]).shape.should eq([2_u32])
      MXNet::NDArray.array([1_u8]).shape.should eq([1_u32])
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

  describe "#reshape" do
    it "reshapes the input array" do
      a = MXNet::NDArray.array([[1.0, 2.0], [3.0, 4.0]])
      a.reshape(shape: [4]).should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape([4]).should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
      a.reshape(4).should eq(MXNet::NDArray.array([1.0, 2.0, 3.0, 4.0]))
    end

    it "supports special values for dimensions" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.reshape([-1, 0], reverse: false).shape.should eq([2_u32, 4_u32])
      c.reshape([-1, 0], reverse: true).shape.should eq([4_u32, 2_u32])
    end
  end

  describe "#flatten" do
    it "flattens the input array" do
      c = MXNet::NDArray.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
      c.flatten.shape.should eq([1_u32, 8_u32])
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

    it "reduces dimensionality correctly" do
      x = MXNet::NDArray.array((0...7 * 5 * 3 * 1).to_a).reshape([7, 5, 3, 1])
      x[1].shape.should eq([5_u32, 3_u32, 1_u32])
      x[0..-1, 1].shape.should eq([7_u32, 3_u32, 1_u32])
      x[0..-1, 0..-1, 1].shape.should eq([7_u32, 5_u32, 1_u32])
      x[0..-1, 1, 0..-1].shape.should eq([7_u32, 3_u32, 1_u32])
      x[0..-1, 1, 1].shape.should eq([7_u32, 1_u32])
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
