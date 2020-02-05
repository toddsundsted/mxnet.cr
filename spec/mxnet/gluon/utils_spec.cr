require "../../spec_helper"

describe MXNet::Gluon::Utils do
  describe ".check_sha1" do
    text = "abcdbcdecdefdefgefghfghighijhi"

    it "checks the hash of the file" do
      File.open(File.join(Dir.tempdir, "foo_bar.txt"), "w") { |io| io.puts text }
      MXNet::Gluon::Utils.check_sha1(File.join(Dir.tempdir, "foo_bar.txt"), "59de2d416061286c1d98cb77b64443f782c45c06").should be_true
    end
  end

  describe ".download" do
    url = "https://raw.githubusercontent.com/toddsundsted/mxnet.cr/master/README.md"
    filename = File.join(Dir.tempdir, "README.md")

    it "returns the path to the downloaded file" do
      MXNet::Gluon::Utils.download(url, Dir.tempdir).should eq(filename)
    ensure
      File.exists?(filename) && File.delete(filename)
    end

    it "downloads the file from the specified URL" do
      File.exists?(MXNet::Gluon::Utils.download(url, Dir.tempdir)).should be_true
    ensure
      File.exists?(filename) && File.delete(filename)
    end

    it "does not download the file if it already exists" do
      File.open(filename, "w") { |io| io.puts "Test Test Test" }
      MXNet::Gluon::Utils.download(url, Dir.tempdir, overwrite: false)
      File.open(filename) { |io| io.gets }.should eq("Test Test Test")
    ensure
      File.exists?(filename) && File.delete(filename)
    end

    it "overwrites the file if forced to do so" do
      File.open(filename, "w") { |io| io.puts "Test Test Test" }
      MXNet::Gluon::Utils.download(url, Dir.tempdir, overwrite: true)
      File.open(filename) { |io| io.gets }.should eq("# Deep Learning for Crystal")
    ensure
      File.exists?(filename) && File.delete(filename)
    end

    it "overwrites the file if the hash doesn't match" do
      File.open(filename, "w") { |io| io.puts "Test Test Test" }
      MXNet::Gluon::Utils.download(url, Dir.tempdir, sha1_hash: "0000000000000000000000000000000000000000")
      File.open(filename) { |io| io.gets }.should eq("# Deep Learning for Crystal")
    ensure
      File.exists?(filename) && File.delete(filename)
    end
  end
end
