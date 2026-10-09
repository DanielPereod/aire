# frozen_string_literal: true

require 'json'
require 'fileutils'

module Aire
  # Escribe una exportación aire-scene v1 (ver backend/aire_backend/scene.py).
  #
  # Ruby puro, sin API de SketchUp, para poder probarlo fuera de SketchUp. Los
  # buffers se escriben en ficheros parciales mientras se recorre el modelo, de
  # modo que la memoria no crece con el número de triángulos.
  class SceneWriter
    FORMAT = 'aire-scene'
    VERSION = 1

    # nombre, dtype numpy, directiva de pack, componentes, :tri o :edge
    BUFFERS = [
      [:positions,          '<f4', 'e', 9, :tri],
      [:normals,            '<f4', 'e', 9, :tri],
      [:uvs,                '<f4', 'e', 6, :tri],
      [:tri_object,         '<u4', 'V', 1, :tri],
      [:tri_material_front, '<u4', 'V', 1, :tri],
      [:tri_material_back,  '<u4', 'V', 1, :tri],
      [:edge_positions,     '<f4', 'e', 6, :edge],
      [:edge_object,        '<u4', 'V', 1, :edge],
      [:tri_clip,           '<u4', 'V', 1, :tri],
      [:edge_clip,          '<u4', 'V', 1, :edge]
    ].freeze

    attr_reader :dir, :triangles, :edges

    def initialize(dir)
      @dir = dir
      FileUtils.mkdir_p(dir)
      @triangles = 0
      @edges = 0
      @parts = {}
      BUFFERS.each do |name, *|
        @parts[name] = File.open(part_path(name), 'wb')
      end
    end

    # p, n: [[x,y,z] x3]  uv: [[u,v] x3]
    # clip: índice del conjunto de planos de sección (0 = ninguno)
    def add_triangle(p, n, uv, object_id, mat_front, mat_back, clip = 0)
      @parts[:positions].write(p.flatten.pack('e9'))
      @parts[:normals].write(n.flatten.pack('e9'))
      @parts[:uvs].write(uv.flatten.pack('e6'))
      @parts[:tri_object].write([object_id].pack('V'))
      @parts[:tri_material_front].write([mat_front].pack('V'))
      @parts[:tri_material_back].write([mat_back].pack('V'))
      @parts[:tri_clip].write([clip].pack('V'))
      @triangles += 1
    end

    def add_edge(a, b, object_id, clip = 0)
      @parts[:edge_positions].write((a + b).pack('e6'))
      @parts[:edge_object].write([object_id].pack('V'))
      @parts[:edge_clip].write([clip].pack('V'))
      @edges += 1
    end

    # meta: hash con view, camera, materials, objects, sun, source...
    def finish(meta)
      @parts.each_value(&:close)
      buffers = []
      offset = 0
      File.open(File.join(dir, 'geometry.bin'), 'wb') do |out|
        BUFFERS.each do |name, dtype, _pack, comps, kind|
          count = comps * (kind == :tri ? @triangles : @edges)
          path = part_path(name)
          File.open(path, 'rb') { |f| IO.copy_stream(f, out) }
          File.delete(path)
          buffers << { name: name.to_s, dtype: dtype, offset: offset, count: count }
          offset += count * 4
        end
      end
      doc = meta.merge(
        format: FORMAT, version: VERSION, units: 'm',
        geometry: { file: 'geometry.bin', triangles: @triangles, edges: @edges, buffers: buffers }
      )
      File.write(File.join(dir, 'scene.json'), JSON.pretty_generate(doc), mode: 'w:UTF-8')
      dir
    end

    def abort
      @parts.each_value { |f| f.close unless f.closed? }
      @parts.each_key { |name| File.delete(part_path(name)) if File.exist?(part_path(name)) }
    end

    private

    def part_path(name)
      File.join(dir, "geometry.#{name}.part")
    end
  end
end
