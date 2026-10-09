# frozen_string_literal: true

# AIRE: render IA para SketchUp. Cargador de la extensión.
require 'sketchup.rb'
require 'extensions.rb'

module Aire
  VERSION = '0.4.0'

  unless file_loaded?(__FILE__)
    ext = SketchupExtension.new('AIRE · Render IA', File.join(__dir__, 'aire', 'main'))
    ext.description = 'Render fotorrealista con IA a partir del modelo 3D (prototipo).'
    ext.version = VERSION
    ext.creator = 'AIRE'
    Sketchup.register_extension(ext, true)
    file_loaded(__FILE__)
  end
end
