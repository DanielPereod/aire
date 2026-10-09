# frozen_string_literal: true

# Stub mínimo de la API de SketchUp para ejecutar el exportador fuera de SketchUp.
# Solo implementa lo que usa Aire::Exporter; no pretende ser fiel en todo lo demás.

module Geom
  class Point3d
    attr_reader :x, :y, :z

    def initialize(x, y, z)
      @x = x.to_f
      @y = y.to_f
      @z = z.to_f
    end

    def to_a = [x, y, z]
    def transform(t) = t.apply_point(self)
  end

  class Vector3d < Point3d; end

  class Transformation
    attr_reader :m # 4x4 por filas

    def initialize(m = nil)
      @m = m || [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]].map { |r| r.map(&:to_f) }
    end

    def self.translation(x, y, z)
      new([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]].map { |r| r.map(&:to_f) })
    end

    def self.scaling(sx, sy, sz)
      new([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]].map { |r| r.map(&:to_f) })
    end

    def *(other)
      o = other.m
      Transformation.new(Array.new(4) { |i| Array.new(4) { |j| (0..3).sum { |k| @m[i][k] * o[k][j] } } })
    end

    def to_a = (0..3).flat_map { |c| (0..3).map { |r| @m[r][c] } }

    def apply_point(p)
      v = [p.x, p.y, p.z, 1.0]
      r = (0..2).map { |i| (0..3).sum { |k| @m[i][k] * v[k] } }
      Point3d.new(*r)
    end
  end

  class BoundingBox
    def initialize(pts)
      @min = (0..2).map { |i| pts.map { |p| p.to_a[i] }.min }
      @max = (0..2).map { |i| pts.map { |p| p.to_a[i] }.max }
    end

    def corner(i)
      Point3d.new(i & 1 == 0 ? @min[0] : @max[0], i & 2 == 0 ? @min[1] : @max[1], i & 4 == 0 ? @min[2] : @max[2])
    end
  end
end

module Sketchup
  def self.status_text=(_txt); end
  def self.version = 'stub'

  Color = Struct.new(:red, :green, :blue)

  class Layer
    attr_reader :name, :folder

    def initialize(name, visible = true)
      @name = name
      @visible = visible
      @folder = nil
    end

    def visible? = @visible
  end

  UNTAGGED = Layer.new('Untagged')

  class Texture
    attr_reader :filename, :width, :height, :image_width, :image_height

    def initialize(filename, width, height)
      @filename = filename
      @width = width
      @height = height
      @image_width = 4
      @image_height = 4
    end

    def write(path, _colorize = false) = File.binwrite(path, 'stub')
  end

  class Material
    attr_reader :name, :color, :texture
    attr_accessor :alpha

    def initialize(name, rgb, texture = nil)
      @name = name
      @color = Color.new(*rgb)
      @texture = texture
      @alpha = 1.0
    end

    def display_name = @name
  end

  class Entity
    attr_accessor :layer, :hidden

    def hidden? = !!@hidden
    def layer = @layer || UNTAGGED
  end

  class PolygonMesh
    def initialize(points, normal)
      @points = points
      @normal = normal
    end

    def count_points = @points.size
    def point_at(i) = @points[i - 1]
    def normal_at(_i) = @normal
    # UV "en pulgadas" en el plano XY, como SketchUp con caras sin material propio
    def uv_at(i, _front) = Geom::Point3d.new(@points[i - 1].x, @points[i - 1].y, 1.0)
    def polygons = (1...(@points.size - 1)).map { |k| [1, k + 1, k + 2] }
  end

  class Face < Entity
    attr_accessor :material, :back_material

    def initialize(pts, material: nil, back_material: nil)
      @pts = pts.map { |p| Geom::Point3d.new(*p) }
      @material = material
      @back_material = back_material
    end

    def mesh(_flags)
      a, b, c = @pts
      u = [b.x - a.x, b.y - a.y, b.z - a.z]
      v = [c.x - a.x, c.y - a.y, c.z - a.z]
      n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
      len = Math.sqrt(n.sum { |x| x * x })
      PolygonMesh.new(@pts, Geom::Vector3d.new(*n.map { |x| x / len }))
    end
  end

  Vertex = Struct.new(:position)

  # Colección de entidades con plano de sección activo opcional
  class Entities < Array
    attr_accessor :active_section_plane
  end

  class SectionPlane < Entity
    def initialize(plane)
      @plane = plane
    end

    def get_plane = @plane
  end

  class Edge < Entity
    attr_reader :start, :end

    def initialize(a, b, soft: false, smooth: false)
      @start = Vertex.new(Geom::Point3d.new(*a))
      @end = Vertex.new(Geom::Point3d.new(*b))
      @soft = soft
      @smooth = smooth
    end

    def soft? = @soft
    def smooth? = @smooth
  end

  class ComponentDefinition
    attr_reader :name, :entities

    def initialize(name, entities)
      @name = name
      @entities = entities
    end

    def bounds
      pts = entities.flat_map { |e| e.is_a?(Face) ? e.mesh(0).polygons.flatten.map { |i| e.mesh(0).point_at(i) } : [] }
      pts = [Geom::Point3d.new(0, 0, 0)] if pts.empty?
      Geom::BoundingBox.new(pts)
    end
  end

  class ComponentInstance < Entity
    attr_reader :definition, :transformation, :name
    attr_accessor :material

    def initialize(definition, transformation = Geom::Transformation.new, name: '', material: nil)
      @definition = definition
      @transformation = transformation
      @name = name
      @material = material
    end

    def persistent_id = object_id
  end

  class Group < ComponentInstance; end

  class Camera
    attr_reader :eye, :target, :up, :fov, :height

    def initialize(eye, target, up, fov: 35.0, perspective: true, height: 100.0)
      @eye = Geom::Point3d.new(*eye)
      @target = Geom::Point3d.new(*target)
      @up = Geom::Vector3d.new(*up)
      @fov = fov
      @perspective = perspective
      @height = height
    end

    def perspective? = @perspective
    def aspect_ratio = 0.0
    def fov_is_height? = true
    def is_2d? = false
  end

  class View
    attr_reader :camera, :vpwidth, :vpheight

    def initialize(camera, w, h)
      @camera = camera
      @vpwidth = w
      @vpheight = h
    end

    def write_image(opts) = File.binwrite(opts[:filename], 'stub')
  end

  class Model
    attr_reader :entities, :title, :path, :active_view

    def initialize(entities, view, title: 'Stub')
      @entities = entities
      @active_view = view
      @title = title
      @path = ''
    end

    def active_path = nil
    def rendering_options = { 'DisplaySectionCuts' => true }
    def shadow_info = { 'SunDirection' => Geom::Vector3d.new(0, 0, 1), 'ShadowTime' => Time.at(0) }
  end
end
