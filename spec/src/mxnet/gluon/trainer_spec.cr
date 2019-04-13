require "../../../spec_helper"

create_double(MXNet::Optimizer)

describe MXNet::Gluon::Trainer do
  describe "#step" do
    p = MXNet::Gluon::Parameter.new("p", shape: [1]).tap do |param|
      param.init
    end
    q = MXNet::Gluon::Parameter.new("q", shape: [1]).tap do |param|
      param.init
    end

    optimizer = instance_double(MXNet::Optimizer)

    it "calls Optimizer#update for every parameter" do
      trainer = MXNet::Gluon::Trainer.new({p: p, q: q}, optimizer)
      trainer.step(1)
      verify(optimizer, update: 2.times)
    end
  end

  describe "#update" do
    p = MXNet::Gluon::Parameter.new("p", shape: [1]).tap do |param|
      param.init
    end
    q = MXNet::Gluon::Parameter.new("q", shape: [1]).tap do |param|
      param.init
    end

    optimizer = instance_double(MXNet::Optimizer)

    it "calls Optimizer#update for every parameter" do
      trainer = MXNet::Gluon::Trainer.new({p: p, q: q}, optimizer)
      trainer.update(1)
      verify(optimizer, update: 2.times)
    end
  end
end
