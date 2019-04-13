macro create_double(clazz, *defs)
  {%
    class_defs = {} of MacroId => ASTNode
    instance_defs = {} of MacroId => ASTNode
  %}

  {% for d in defs %}
    {% if ["self", clazz.stringify].includes?(d.receiver.stringify) %}
      {% class_defs[d.name] = d.body %}
    {% elsif !d.receiver %}
      {% instance_defs[d.name] = d.body %}
    {% else %}
      {% raise "invalid receiver: use either `#{clazz}` or `self`" %}
    {% end %}
  {% end %}

  class {{"#{clazz}Double__".id}} < {{clazz}}
    @@__counts = Hash(::Symbol, Int32).new { 0 }

    def self.__count(name)
      @@__counts[name]? || 0
    end

    @__counts = Hash(::Symbol, Int32).new { 0 }

    def __count(name)
      @__counts[name]? || 0
    end

    {% for method in clazz.resolve.class.methods %}
      {% unless method.name == "new" || method.name == "allocate" %}
        {% type_vars = [] of String %}
        {% for arg in method.args %}
          {% if arg.restriction.is_a?(Generic) %}
            {% type_vars = type_vars + arg.restriction.type_vars.map(&.stringify) %}
          {% elsif arg.restriction.is_a?(Path) && !arg.restriction.resolve? %}
            {% type_vars = type_vars << arg.restriction.stringify %}
          {% end %}
        {% end %}
        {% if type_vars.size > 0 %}
          def self.{{method.name}}({{*method.args}}){{(" forall " + type_vars.sort.uniq.join(",")).id}}
        {% else %}
          def self.{{method.name}}({{*method.args}})
        {% end %}
          @@__counts[:{{method.name.stringify}}] += 1
          {% if class_defs[method.name] %}
            {{class_defs[method.name]}}
          {% elsif !method.body.is_a?(Nop) %}
            {{method.body}}
          {% end %}
        end
      {% end %}
    {% end %}

    {% for method in clazz.resolve.methods %}
      {% unless method.name == "initialize" %}
        {% type_vars = [] of String %}
        {% for arg in method.args %}
          {% if arg.restriction.is_a?(Generic) %}
            {% type_vars = type_vars + arg.restriction.type_vars.map(&.stringify) %}
          {% elsif arg.restriction.is_a?(Path) && !arg.restriction.resolve? %}
            {% type_vars = type_vars << arg.restriction.stringify %}
          {% end %}
        {% end %}
        {% if type_vars.size > 0 %}
          def {{method.name}}({{*method.args}}){{(" forall " + type_vars.sort.uniq.join(",")).id}}
        {% else %}
          def {{method.name}}({{*method.args}})
        {% end %}
          @__counts[:{{method.name.stringify}}] += 1
          {% if instance_defs[method.name] %}
            {{instance_defs[method.name]}}
          {% elsif !method.body.is_a?(Nop) %}
            {{method.body}}
          {% end %}
        end
      {% end %}
    {% end %}
  end
end

macro class_double(clazz)
  {{"#{clazz}Double__".id}}
end

macro instance_double(clazz)
  {{"#{clazz}Double__".id}}.new
end

def verify(double, file = __FILE__, line = __LINE__, **calls)
  raise "specify the method calls to verify" if calls.empty?
  calls.each do |name, expected|
    expected = expected.size if expected.responds_to?(:size)
    unless (actual = double.__count(name)) == expected
      counter = expected == 1 ? "time" : "times"
      raise Spec::AssertionFailed.new(
        "Expected '#{name}' to be called #{expected} #{counter}, instead of #{actual}",
        file, line
      )
    end
  end
end

def once
  1
end

def twice
  2
end
