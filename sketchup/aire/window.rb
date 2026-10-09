# frozen_string_literal: true

require 'json'
require 'base64'
require 'fileutils'
require_relative 'runner'
require_relative 'exporter'

module Aire
  # Ventana principal de AIRE (HtmlDialog). Pensada para alguien no técnico:
  # preparar AIRE una vez, elegir estilo y luz, y pulsar "Renderizar esta vista".
  class Window
    STALE_SECONDS = 15 * 60 # sin noticias de una tarea en marcha = se interrumpió
    WIDTHS = { 'rapida' => 1024, 'alta' => 1536, 'comparar' => 1024 }.freeze

    def self.show
      @instance ||= new
      @instance.show
    end

    def initialize
      @dialog = UI::HtmlDialog.new(
        dialog_title: 'AIRE · Render con IA', preferences_key: 'AIRE_window',
        width: 460, height: 820, min_width: 380, min_height: 560,
        style: UI::HtmlDialog::STYLE_DIALOG
      )
      @dialog.set_file(File.join(__dir__, 'ui', 'index.html'))
      @job_dir = nil
      @timer = nil
      register_callbacks
    end

    def show
      @dialog.visible? ? @dialog.bring_to_front : @dialog.show
    end

    private

    def register_callbacks
      @dialog.add_action_callback('ready') { |_ctx| push_state(history: true) }
      @dialog.add_action_callback('install') { |_ctx| start_install }
      @dialog.add_action_callback('render') { |_ctx, json| start_render(JSON.parse(json)) }
      @dialog.add_action_callback('open_image') { |_ctx, path| open_path(path) }
      @dialog.add_action_callback('open_folder') { |_ctx, path| open_path(File.dirname(path)) }
      @dialog.add_action_callback('open_log') { |_ctx, path| open_path(path) }
      @dialog.add_action_callback('save_image') { |_ctx, path| save_image(path) }
      @dialog.set_on_closed { stop_timer }
    end

    # ------------------------------------------------------------------ estado
    def push_state(history: false)
      state = {
        ready: Env.ready?,
        config: Env.config.slice('gpu', 'vram_gb', 'ram_gb', 'preset_title'),
        setup: task_state(Env.setup_progress_file),
        job: @job_dir ? task_state(File.join(@job_dir, 'job.json')) : nil
      }
      state[:history] = history_items if history
      js("aire.setState(#{JSON.generate(state)})")
      ensure_timer if running?(state[:setup]) || running?(state[:job])
    end

    def task_state(path)
      s = Env.read_json(path)
      return nil unless s

      if s['state'] == 'running'
        age = Time.now.to_f - s['updated'].to_f
        # Se interrumpió si no da señales o si su proceso ya no existe (se cerró sin avisar).
        # El margen de 20 s cubre el relevo entre bootstrap.ps1 y el instalador en Python.
        dead = age > 20 && !Env.process_alive?(s['pid'])
        s['state'] = 'interrupted' if age > STALE_SECONDS || dead
      end
      s
    end

    def running?(s)
      s && s['state'] == 'running'
    end

    def ensure_timer
      return if @timer

      @timer = UI.start_timer(1.0, true) { tick }
    end

    def stop_timer
      UI.stop_timer(@timer) if @timer
      @timer = nil
    end

    def tick
      setup = task_state(Env.setup_progress_file)
      job = @job_dir ? task_state(File.join(@job_dir, 'job.json')) : nil
      finished = !running?(setup) && !running?(job)
      push_state(history: finished && job && job['state'] == 'done')
      stop_timer if finished
    end

    # ------------------------------------------------------------------ acciones
    def start_install
      return if running?(task_state(Env.setup_progress_file))

      file = Env.setup_progress_file
      FileUtils.mkdir_p(File.dirname(file))
      File.write(file, JSON.generate(state: 'running', steps: ['Preparando el instalador'], step: 0,
                                     title: 'Preparando el instalador', detail: 'Empezando…',
                                     updated: Time.now.to_f))
      Runner.bootstrap(file)
      push_state
    end

    def start_render(params)
      return notify('AIRE todavía no está preparado.') unless Env.ready?
      return if @job_dir && running?(task_state(File.join(@job_dir, 'job.json')))

      quality = WIDTHS.key?(params['quality']) ? params['quality'] : 'alta'
      job_dir = File.join(Env.home, 'jobs', Time.now.strftime('%Y%m%d-%H%M%S'))
      export_dir = File.join(job_dir, 'export')
      model = Sketchup.active_model
      result = Exporter.new(model, model.active_view, width: WIDTHS[quality]).export(export_dir)
      if result[:triangles].zero?
        FileUtils.rm_rf(job_dir)
        return notify('No hay nada visible en esta vista. Encuadra la habitación y vuelve a probar.')
      end

      prompt_file = File.join(job_dir, 'prompt.txt')
      File.write(prompt_file, params['prompt'].to_s, mode: 'w:UTF-8')
      progress = File.join(job_dir, 'job.json')
      File.write(progress, JSON.generate(state: 'running', steps: ['Preparando la escena'], step: 0,
                                         title: 'Preparando la escena', detail: '', updated: Time.now.to_f))
      Runner.python('job', ['--home', Env.home, '--export', export_dir, '--style', params['style'].to_s,
                            '--light', params['light'].to_s, '--quality', quality,
                            '--variants', params['variants'].to_i.clamp(1, 4).to_s,
                            '--prompt-file', prompt_file, '--progress', progress])
      @job_dir = job_dir
      push_state
    rescue StandardError => e
      notify("No se ha podido leer el modelo: #{e.message}")
    end

    def save_image(path)
      return unless File.exist?(path)

      dest = UI.savepanel('Guardar imagen', Dir.home, "AIRE-#{File.basename(File.dirname(File.dirname(path)))}.png")
      return unless dest

      dest += '.png' unless dest.downcase.end_with?('.png')
      FileUtils.cp(path, dest)
      js('aire.toast("Imagen guardada")')
    end

    def open_path(path)
      UI.openURL("file:///#{path.tr('\\', '/')}") if path && File.exist?(path)
    end

    # ------------------------------------------------------------------ galería
    def history_items(limit = 24)
      dirs = Dir.glob(File.join(Env.home, 'jobs', '*')).sort.reverse
      items = []
      dirs.each do |dir|
        res = Env.read_json(File.join(dir, 'renders', 'result.json'))
        next unless res

        res['images'].each do |img|
          thumb = img['thumb'] && File.exist?(img['thumb']) ? Base64.strict_encode64(File.binread(img['thumb'])) : nil
          items << { path: img['path'], thumb: thumb && "data:image/jpeg;base64,#{thumb}",
                     style: res['style'], light: res['light'], text: res['user_prompt'],
                     created: res['created'], label: img['label'] }
        end
        break if items.size >= limit
      end
      items.first(limit)
    end

    # Aviso breve en la ventana; también restablece el botón de crear imagen
    def notify(text)
      js("aire.toast(#{JSON.generate(text)})")
      push_state
    end

    def js(code)
      @dialog.execute_script(code)
    end
  end
end
