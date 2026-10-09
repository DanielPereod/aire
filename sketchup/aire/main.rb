# frozen_string_literal: true

require 'sketchup.rb'
require_relative 'exporter'

module Aire
  def self.export_scene
    model = Sketchup.active_model
    base = UI.select_directory(title: 'AIRE: carpeta donde guardar la exportación')
    return unless base

    width = Sketchup.read_default('AIRE', 'export_width', 1536).to_i
    input = UI.inputbox(['Ancho de salida (px)'], [width], 'AIRE: exportar escena')
    return unless input

    width = input[0].to_i.clamp(256, 8192)
    Sketchup.write_default('AIRE', 'export_width', width)

    stamp = Time.now.strftime('%Y%m%d-%H%M%S')
    title = model.title.to_s.empty? ? 'modelo' : model.title.gsub(/[^0-9A-Za-z_-]+/, '_')
    dir = File.join(base, "#{title}-#{stamp}")

    t0 = Time.now
    result = Exporter.new(model, model.active_view, width: width).export(dir)
    secs = (Time.now - t0).round(1)

    msg = +"Exportado en #{secs}s:\n#{dir}\n\n" \
           "#{result[:triangles]} triángulos · #{result[:edges]} aristas · " \
           "#{result[:objects]} objetos · #{result[:materials]} materiales\n\n" \
           "Genera los pases con:\npython -m aire_backend.passes \"#{dir}\""
    msg << "\n\nAvisos:\n- #{result[:warnings].join("\n- ")}" unless result[:warnings].empty?
    UI.messagebox(msg)
  rescue StandardError => e
    UI.messagebox("AIRE: error al exportar\n\n#{e.message}")
    puts e.full_message
  end

  unless file_loaded?(__FILE__)
    menu = UI.menu('Extensions').add_submenu('AIRE')
    menu.add_item('Exportar escena para render (P1)…') { export_scene }
    file_loaded(__FILE__)
  end
end
