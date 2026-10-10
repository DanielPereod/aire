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
    WIDTHS = { 'rapida' => 1024, 'alta' => 1920, 'real' => 1920, 'comparar' => 1920 }.freeze

    def self.show
      @instance ||= new
      @instance.show
    end

    def initialize
      @dialog = UI::HtmlDialog.new(
        dialog_title: 'AIRE · Render con IA', preferences_key: 'AIRE_window',
        width: 560, height: 900, min_width: 420, min_height: 600,
        style: UI::HtmlDialog::STYLE_DIALOG
      )
      @dialog.set_file(File.join(__dir__, 'ui', 'index.html'))
      @job_dir = nil
      @job_kind = nil
      @timer = nil
      register_callbacks
    end

    def show
      @dialog.visible? ? @dialog.bring_to_front : @dialog.show
    end

    private

    def register_callbacks
      on('ready') { |_ctx| push_state(history: true) }
      on('install') { |_ctx| start_install }
      on('render') { |_ctx, json| start_render(JSON.parse(json)) }
      on('open_image') { |_ctx, path| open_path(path) }
      on('open_folder') { |_ctx, path| open_path(File.dirname(path)) }
      on('open_log') { |_ctx, path| open_path(path) }
      on('save_image') { |_ctx, path| save_image(path) }
      on('open_jobs') { |_ctx| open_jobs }
      on('edit_load') { |_ctx, path| edit_load(path) }
      on('pick_reference') { |_ctx| pick_reference }
      on('edit') { |_ctx, json| start_edit(JSON.parse(json)) }
      on('enhance') { |_ctx, path| start_enhance(path.to_s) }
      @dialog.set_on_closed { stop_timer }
    end

    # Ningún error de Ruby debe dejar la ventana en blanco: se registra y se muestra.
    def on(name, &block)
      @dialog.add_action_callback(name) do |*args|
        block.call(*args)
      rescue StandardError, ScriptError => e
        report_error(e, name)
      end
    end

    def report_error(error, where)
      log = File.join(Env.home, 'logs', 'ui.log')
      begin
        FileUtils.mkdir_p(File.dirname(log))
        File.open(log, 'a:UTF-8') do |f|
          f.puts "=== #{Time.now} [#{where}] AIRE #{Aire::VERSION} SketchUp #{Sketchup.version}"
          f.puts "#{error.class}: #{error.message}"
          f.puts Array(error.backtrace).first(25)
        end
      rescue StandardError
        nil
      end
      puts "AIRE [#{where}] #{error.class}: #{error.message}"
      msg = "#{error.class}: #{error.message}".encode('UTF-8', invalid: :replace, undef: :replace)
      js("aire.fatal(#{JSON.generate(msg)}, #{JSON.generate(log)})")
    rescue StandardError
      nil
    end

    # ------------------------------------------------------------------ estado
    def push_state(history: false)
      state = {
        ready: Env.ready?,
        config: Env.config.slice('gpu', 'vram_gb', 'ram_gb', 'preset_title'),
        setup: task_state(Env.setup_progress_file),
        job: @job_dir ? task_state(File.join(@job_dir, 'job.json')) : nil,
        job_kind: @job_kind
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
      tick!
    rescue StandardError => e
      stop_timer
      report_error(e, 'tick')
    end

    def tick!
      setup = task_state(Env.setup_progress_file)
      job = @job_dir ? task_state(File.join(@job_dir, 'job.json')) : nil
      finished = !running?(setup) && !running?(job)
      done = finished && job && job['state'] == 'done'
      push_state(history: done)
      stop_timer if finished
      edit_finished(job) if done && @job_kind == 'edit'
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
      @job_kind = 'render'
      push_state
    rescue StandardError => e
      notify("No se ha podido leer el modelo: #{e.message}")
    end

    # ------------------------------------------------------------------ edición
    IMAGE_TYPES = { '.png' => 'image/png', '.jpg' => 'image/jpeg', '.jpeg' => 'image/jpeg',
                    '.webp' => 'image/webp' }.freeze
    MAX_REFERENCE_BYTES = 25 * 1024 * 1024

    def data_url(path)
      type = IMAGE_TYPES[File.extname(path).downcase] || 'image/png'
      "data:#{type};base64,#{Base64.strict_encode64(File.binread(path))}"
    end

    # Vista previa ligera (jpg de 1600 px) si existe; si no, la imagen original
    def preview_of(path)
      res = Env.read_json(File.join(File.dirname(path), 'result.json'))
      item = res && res['images'].to_a.find { |i| File.expand_path(i['path'].to_s) == File.expand_path(path) }
      prev = item && item['preview']
      prev && File.exist?(prev) ? prev : path
    end

    def edit_load(path)
      return notify('No se encuentra esa imagen.') unless path && File.exist?(path)

      res = Env.read_json(File.join(File.dirname(path), 'result.json')) || {}
      source = res['source'] && File.exist?(res['source']) ? data_url(preview_of(res['source'])) : nil
      data = { path: path, src: data_url(preview_of(path)), source: source }
      js("aire.editImage(#{JSON.generate(data)})")
    end

    def pick_reference
      dir = Sketchup.read_default('AIRE', 'reference_dir', Dir.home)
      path = UI.openpanel('Elige una imagen de referencia', dir, 'Imágenes|*.png;*.jpg;*.jpeg;*.webp||')
      return unless path && File.exist?(path)

      Sketchup.write_default('AIRE', 'reference_dir', File.dirname(path))
      return notify('Esa imagen es demasiado grande (más de 25 MB).') if File.size(path) > MAX_REFERENCE_BYTES
      return notify('Usa una imagen PNG, JPG o WEBP.') unless IMAGE_TYPES.key?(File.extname(path).downcase)

      ref = { path: path, name: File.basename(path), src: data_url(path) }
      js("aire.addReference(#{JSON.generate(ref)})")
    end

    def start_edit(params)
      return notify('AIRE todavía no está preparado.') unless Env.ready?
      return notify('Espera a que termine la imagen en curso.') if @job_dir && running?(task_state(File.join(@job_dir, 'job.json')))

      image = params['path'].to_s
      return notify('No se encuentra esa imagen.') unless File.exist?(image)

      job_dir = File.join(Env.home, 'jobs', "#{Time.now.strftime('%Y%m%d-%H%M%S')}-cambio")
      FileUtils.mkdir_p(job_dir)
      prompt_file = File.join(job_dir, 'cambio.txt')
      File.write(prompt_file, params['prompt'].to_s, mode: 'w:UTF-8')
      args = ['--home', Env.home, '--image', image, '--job-dir', job_dir, '--prompt-file', prompt_file]
      mask = params['mask'].to_s
      if mask.start_with?('data:image/png;base64,')
        mask_file = File.join(job_dir, 'zona.png')
        File.binwrite(mask_file, Base64.decode64(mask.split(',', 2)[1]))
        args += ['--mask', mask_file]
      end
      Array(params['refs']).first(3).each { |r| args += ['--ref', r.to_s] if File.exist?(r.to_s) }
      progress = File.join(job_dir, 'job.json')
      File.write(progress, JSON.generate(state: 'running', steps: ['Preparando la imagen'], step: 0,
                                         title: 'Preparando la imagen', detail: '', updated: Time.now.to_f))
      Runner.python('edit', args + ['--progress', progress])
      @job_dir = job_dir
      @job_kind = 'edit'
      push_state
    end

    # «Procesar con IA» sobre una imagen ya creada: sale como imagen nueva en la galería
    def start_enhance(image)
      return notify('AIRE todavía no está preparado.') unless Env.ready?
      return notify('Espera a que termine la imagen en curso.') if @job_dir && running?(task_state(File.join(@job_dir, 'job.json')))
      return notify('No se encuentra esa imagen.') unless File.exist?(image)

      job_dir = File.join(Env.home, 'jobs', "#{Time.now.strftime('%Y%m%d-%H%M%S')}-ia")
      FileUtils.mkdir_p(job_dir)
      progress = File.join(job_dir, 'job.json')
      File.write(progress, JSON.generate(state: 'running', steps: ['Preparando la imagen'], step: 0,
                                         title: 'Preparando la imagen', detail: '', updated: Time.now.to_f))
      Runner.python('enhance', ['--home', Env.home, '--image', image, '--job-dir', job_dir, '--progress', progress])
      @job_dir = job_dir
      @job_kind = 'render'
      push_state
    end

    def edit_finished(job)
      path = job.dig('result', 'images', 0, 'path')
      js("aire.editDone(#{JSON.generate({ path: path })})") if path
    end

    def save_image(path)
      return unless File.exist?(path)

      dest = UI.savepanel('Guardar imagen', Dir.home, "AIRE-#{File.basename(File.dirname(File.dirname(path)))}.png")
      return unless dest

      dest += '.png' unless dest.downcase.end_with?('.png')
      FileUtils.cp(path, dest)
      js('aire.toast("Imagen guardada")')
    end

    def open_jobs
      dir = File.join(Env.home, 'jobs')
      FileUtils.mkdir_p(dir)
      open_path(dir)
    end

    def open_path(path)
      UI.openURL("file:///#{path.tr('\\', '/')}") if path && File.exist?(path)
    end

    # ------------------------------------------------------------------ galería
    def history_items(limit = 24)
      # Dir.glob trata «\» como escape: con C:\Users\... no encuentra nada en Windows
      dirs = Dir.glob(File.join(Env.home.tr('\\', '/'), 'jobs', '*')).sort.reverse
      items = []
      dirs.each do |dir|
        res = Env.read_json(File.join(dir, 'renders', 'result.json'))
        next unless res.is_a?(Hash)

        # un trabajo raro (copiado, a medias o de otra versión) no debe dejar vacía la galería
        begin
          Array(res['images']).each do |img|
            next unless img.is_a?(Hash) && img['path'] && File.exist?(img['path'])

            thumb = img['thumb'] && File.exist?(img['thumb']) ? Base64.strict_encode64(File.binread(img['thumb'])) : nil
            items << { path: img['path'], thumb: thumb && "data:image/jpeg;base64,#{thumb}",
                       style: res['style'], light: res['light'], text: res['user_prompt'],
                       created: res['created'], label: img['label'], quality: res['quality'],
                       kind: res['kind'] || 'render', source: res['source'] }
          end
        rescue StandardError
          next
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
