# frozen_string_literal: true

require 'json'

module Aire
  # Rutas de AIRE y lanzamiento de procesos en segundo plano, sin ventanas.
  module Env
    module_function

    def windows?
      RUBY_PLATFORM =~ /mswin|mingw|cygwin/ ? true : false
    end

    # Todo lo que instala AIRE vive aquí; desinstalar es borrar esta carpeta.
    def home
      base = ENV['LOCALAPPDATA'] || File.join(Dir.home, '.local', 'share')
      File.join(base, 'AIRE')
    end

    def backend_dir
      File.join(__dir__, 'backend')
    end

    def run_py
      File.join(backend_dir, 'run.py')
    end

    # pythonw.exe no abre consola en Windows
    def python
      windows? ? File.join(home, 'env', 'Scripts', 'pythonw.exe') : File.join(home, 'env', 'bin', 'python')
    end

    def config
      read_json(File.join(home, 'config.json')) || {}
    end

    def ready?
      config['ready'] == true && File.exist?(python)
    end

    def setup_progress_file
      File.join(home, 'progress', 'setup.json')
    end

    def read_json(path)
      return nil unless File.exist?(path)

      JSON.parse(File.read(path, encoding: 'UTF-8'))
    rescue JSON::ParserError, SystemCallError
      nil # el fichero se está escribiendo justo ahora; se leerá en el siguiente tick
    end
  end

  module Runner
    module_function

    def quote(arg)
      s = arg.to_s
      s =~ /[\s"]/ ? "\"#{s.gsub('"', '\\"')}\"" : s
    end

    # Lanza un proceso desacoplado de SketchUp y sin ventana. No espera.
    def spawn(exe, args)
      if Env.windows?
        require 'win32ole'
        WIN32OLE.codepage = WIN32OLE::CP_UTF8 # rutas con acentos o ñ
        cmd = ([exe] + args).map { |a| quote(a) }.join(' ')
        WIN32OLE.new('WScript.Shell').Run(cmd, 0, false) # 0 = oculto, false = no esperar
      else
        pid = Process.spawn(exe.to_s, *args.map(&:to_s), %i[out err] => File::NULL)
        Process.detach(pid)
      end
    end

    def bootstrap(progress_file)
      script = File.join(__dir__, 'bin', 'bootstrap.ps1')
      args = ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden', '-File', script,
              '-AireHome', Env.home, '-Backend', Env.backend_dir, '-Progress', progress_file]
      spawn('powershell.exe', args)
    end

    def python(module_name, args)
      spawn(Env.python, [Env.run_py, module_name] + args)
    end
  end
end
