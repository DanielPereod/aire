# frozen_string_literal: true

# Exporta una escena de prueba con el stub de SketchUp:
#   ruby sketchup/test/export_stub_scene.rb <carpeta_salida>
# La verifica backend/tests/test_ruby_export.py.

require_relative 'sketchup_stub'
module Aire
  VERSION = 'test'
end
require_relative '../aire/exporter'

out = ARGV.fetch(0)
IN = 1 / 0.0254 # 1 metro en pulgadas

red = Sketchup::Material.new('Rojo', [200, 20, 20])
wood = Sketchup::Material.new('Madera', [120, 80, 50], Sketchup::Texture.new('roble.jpg', 0.5 * IN, 0.25 * IN))

# Cuadrado unidad (1 m) en el plano XY mirando a +Z, con sus aristas
square = lambda do |material: nil|
  pts = [[0, 0, 0], [IN, 0, 0], [IN, IN, 0], [0, IN, 0]]
  [Sketchup::Face.new(pts, material: material)] +
    pts.each_index.map { |i| Sketchup::Edge.new(pts[i], pts[(i + 1) % 4]) }
end

tile = Sketchup::ComponentDefinition.new('Baldosa', square.call)
red_face = Sketchup::ComponentDefinition.new('Placa roja', square.call(material: red))

hidden_tag = Sketchup::Layer.new('Oculta', false)
hidden_inst = Sketchup::ComponentInstance.new(tile, Geom::Transformation.translation(0, 0, 5 * IN), name: 'Invisible')
hidden_inst.layer = hidden_tag

inner = Sketchup::ComponentInstance.new(tile, Geom::Transformation.translation(2 * IN, 0, 0), name: 'Hija')
parent_ents = Sketchup::Entities.new([inner, Sketchup::Edge.new([0, 0, 0], [0, 0, IN], soft: true)])
# Plano de sección activo en el contexto del padre: x_local = 2,5 m → x_mundo = 5 m (escala x2)
parent_ents.active_section_plane = Sketchup::SectionPlane.new([1.0, 0.0, 0.0, -2.5 * IN])
parent_def = Sketchup::ComponentDefinition.new('Conjunto', parent_ents)

entities = [
  # Instancia heredando la madera (textura 0,5 x 0,25 m)
  Sketchup::ComponentInstance.new(tile, Geom::Transformation.new, name: 'Suelo', material: wood),
  # Espejo en X: la normal debe seguir apuntando a +Z y el orden de vértices corregido
  Sketchup::ComponentInstance.new(red_face, Geom::Transformation.scaling(-1, 1, 1), name: 'Espejo'),
  # Anidado y trasladado 10 m en Y, con escala no uniforme (2, 1, 1)
  Sketchup::Group.new(parent_def, Geom::Transformation.translation(0, 10 * IN, 0) * Geom::Transformation.scaling(2, 1, 1),
                      name: 'Padre'),
  hidden_inst
]

cam = Sketchup::Camera.new([0, -5 * IN, 2 * IN], [0, 0, 0], [0, 0, 1], fov: 50.0)
view = Sketchup::View.new(cam, 1200, 800)
model = Sketchup::Model.new(entities, view, title: 'Stub')
result = Aire::Exporter.new(model, view, width: 600).export(out)
puts result.inspect
