# frozen_string_literal: true

require 'sketchup.rb'
require_relative 'exporter'
require_relative 'runner'
require_relative 'window'

module Aire
  # Exportación manual (para pruebas técnicas, P1)
  def self.export_scene
    model = Sketchup.active_model
    base = UI.select_directory(title: 'AIRE: carpeta donde guardar la exportación')
    return unless base

    width = Sketchup.read_default('AIRE', 'export_width', 1536).to_i
    input = UI.inputbox(['Ancho de salida (px)'], [width], 'AIRE: exportar escena')
    return unless input

    width = input[0].to_i.clamp(256, 8192)
    Sketchup.write_default('AIRE', 'export_width', width)
    title = model.title.to_s.empty? ? 'modelo' : model.title.gsub(/[^0-9A-Za-z_-]+/, '_')
    dir = File.join(base, "#{title}-#{Time.now.strftime('%Y%m%d-%H%M%S')}")
    result = Exporter.new(model, model.active_view, width: width).export(dir)
    msg = +"Exportado:\n#{dir}\n\n#{result[:triangles]} triángulos · #{result[:objects]} objetos"
    msg << "\n\nAvisos:\n- #{result[:warnings].join("\n- ")}" unless result[:warnings].empty?
    UI.messagebox(msg)
  rescue StandardError => e
    UI.messagebox("AIRE: error al exportar\n\n#{e.message}")
  end

  # Al cerrar SketchUp se apaga el motor para liberar la memoria de la tarjeta gráfica
  class QuitObserver < Sketchup::AppObserver
    def onQuit
      return unless Env.ready?

      Runner.python('comfyctl', ['stop', '--home', Env.home])
    rescue StandardError
      nil
    end
  end

  unless file_loaded?(__FILE__)
    menu = UI.menu('Extensions').add_submenu('AIRE')
    menu.add_item('Abrir AIRE…') { Window.show }
    menu.add_separator
    menu.add_item('Exportar escena (técnico)…') { export_scene }

    cmd = UI::Command.new('AIRE') { Window.show }
    icon = File.join(__dir__, 'ui', 'icon.svg')
    cmd.small_icon = icon
    cmd.large_icon = icon
    cmd.tooltip = 'AIRE: render con IA'
    cmd.status_bar_text = 'Crea una imagen realista de la vista actual con inteligencia artificial.'
    toolbar = UI::Toolbar.new('AIRE')
    toolbar.add_item(cmd)
    toolbar.restore

    Sketchup.add_observer(QuitObserver.new)
    file_loaded(__FILE__)
  end
end
