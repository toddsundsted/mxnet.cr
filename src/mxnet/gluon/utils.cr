require "../gluon"
require "digest/sha1"
require "http/client"

module MXNet
  module Gluon
    module Utils
      # Checks whether the SHA1 hash of the file matches the specified
      # hash.
      #
      # ### Parameters
      # * *filename* (`String`)
      #   Path to the file to check.
      # * *sha1_hash* (`String`)
      #   Expected SHA1 hash in hexadecimal.
      #
      def self.check_sha1(filename, sha1_hash)
        File.open(filename) do |io|
          sha1_hash == Digest::SHA1.hexdigest do |digest|
            slice = Bytes.new(1048576)
            while (count = io.read(slice)) > 0
              digest.update(slice[0, count])
            end
          end
        end
      end

      # Downloads from a URL.
      #
      # Returns the file path of the downloaded file.
      #
      # ### Parameters
      # * *url* (`String`)
      #   URL to download.
      # * *path* (`String`, optional)
      #   Destination path to store downloaded file. By default,
      #   stores to the current directory.
      # * *overwrite* (`Bool`, default = `false`)
      #   Whether to overwrite destination file if it already exists.
      # * *sha1_hash* (`String`, optional)
      #   Expected SHA1 hash in hexadecimal. Will overwrite existing
      #   file when hash is specified but doesn't match.
      #
      def self.download(url, path = nil, overwrite = false, sha1_hash = nil)
        if (fname = url.split("/").last).blank?
          raise ArgumentError.new("can't construct file name from this URL: #{url}")
        end
        if path
          path = File.expand_path(path)
          if File.directory?(path)
            fname = File.join(path, fname)
          else
            fname = path
          end
        end
        if overwrite || !File.exists?(fname) || (sha1_hash && !check_sha1(fname, sha1_hash))
          HTTP::Client.get(url) do |response|
            raise "failed to download URL: #{url}" unless response.status_code == 200
            File.open(fname, "w") do |io|
              IO.copy(response.body_io, io)
            end
          end
        end
        fname
      end

      # Returns the base URL for the Gluon dataset and model repository.
      #
      def self.get_repo_url
        default_repo = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/"
        repo_url = ENV.fetch("MXNET_GLUON_REPO", default_repo)
        repo_url += "/" unless repo_url.ends_with?("/")
        repo_url
      end

      # Returns the URL for hosted file in the Gluon repository.
      #
      # ### Parameters
      # * *namespace* (`String`)
      #   Namespace of the file.
      # * *filename* (`String`)
      #   Name of the file.
      #
      def self.get_repo_file_url(namespace, filename)
        "#{get_repo_url}#{namespace}/#{filename}"
      end
    end
  end
end
