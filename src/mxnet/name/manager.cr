module MXNet
  module Name
    # Name manager to do automatic naming.
    #
    class Manager
      @@current = Manager.new

      def self.current
        @@current
      end

      @counter = Hash(String, Int32).new { 0 }

      # Gets the canonical name for a symbol.
      #
      # If *name* is specified, the specified name will be used.
      # Otherwise, automatically generate a unique name based on
      # *hint*.
      #
      # ### Parameters
      # * *name* (`String` or `nil`)
      #   The specified name.
      # * *hint* (`String`)
      #   The hint string.
      #
      def get(name : String?, hint : String)
        return name if name
        name = "#{hint}#{@counter[hint]}"
        @counter[hint] += 1
        name
      end
    end
  end
end
