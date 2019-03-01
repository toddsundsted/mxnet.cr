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

  describe "#to_s" do
    it "pretty-prints the array" do
      MXNet::NDArray.array([1.0, 2.0, 3.0]).to_s.should eq("[1.0, 2.0, 3.0]\n<NDArray 3 float64 cpu(0)>")
      MXNet::NDArray.array([[1_i64, 2_i64], [3_i64, 4_i64]]).to_s.should eq("[[1, 2], [3, 4]]\n<NDArray 2x2 int64 cpu(0)>")
      MXNet::NDArray.array([1, 2]).to_s.should eq("[1, 2]\n<NDArray 2 int32 cpu(0)>")
      MXNet::NDArray.array([1_u8]).to_s.should eq("[1]\n<NDArray 1 uint8 cpu(0)>")
    end
  end
end
