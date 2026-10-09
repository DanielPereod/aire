# frozen_string_literal: true

# Prueba la lógica de la ventana de AIRE fuera de SketchUp:
#   ruby sketchup/test/window_test.rb
require 'minitest/autorun'
require 'tmpdir'
require 'json'
require_relative 'sketchup_stub'

def file_loaded?(_f) = true
def file_loaded(_f); end

module UI
  class HtmlDialog
    STYLE_DIALOG = 1
    attr_reader :callbacks, :scripts, :file

    def initialize(_opts)
      @callbacks = {}
      @scripts = []
    end

    def set_file(path) = (@file = path)
    def add_action_callback(name, &blk) = (@callbacks[name] = blk)
    def set_on_closed(&blk) = (@on_closed = blk)
    def execute_script(code) = @scripts << code
    def visible? = false
    def show; end
    def trigger(name, *args) = @callbacks.fetch(name).call(nil, *args)
  end

  @timers = []
  class << self
    attr_reader :timers

    def start_timer(_secs, _repeat, &blk)
      @timers << blk
      @timers.size
    end

    def stop_timer(_id); end
    def openURL(_url); end
    def savepanel(*) = nil
  end
end

module Sketchup
  class << self
    attr_accessor :active_model
  end
end

module Aire
  VERSION = 'test'
end
require_relative '../aire/window'

class WindowTest < Minitest::Test
  IN = 1 / 0.0254

  def setup
    @tmp = Dir.mktmpdir
    ENV['LOCALAPPDATA'] = @tmp
    @spawned = []
    spawned = @spawned
    Aire::Runner.define_singleton_method(:spawn) { |exe, args| spawned << [exe, args] }
    Aire::Env.define_singleton_method(:windows?) { true }
    pts = [[0, 0, 0], [IN, 0, 0], [IN, IN, 0], [0, IN, 0]]
    cam = Sketchup::Camera.new([0.5 * IN, -3 * IN, 1 * IN], [0.5 * IN, 0, 0], [0, 0, 1], fov: 50.0)
    view = Sketchup::View.new(cam, 800, 600)
    Sketchup.active_model = Sketchup::Model.new([Sketchup::Face.new(pts)], view)
    @win = Aire::Window.new
    @dialog = @win.instance_variable_get(:@dialog)
  end

  def teardown
    FileUtils.rm_rf(@tmp)
  end

  def last_state
    js = @dialog.scripts.reverse.find { |s| s.start_with?('aire.setState(') }
    JSON.parse(js[/\Aaire\.setState\((.*)\)\z/m, 1])
  end

  def make_ready
    FileUtils.mkdir_p(Aire::Env.home)
    File.write(File.join(Aire::Env.home, 'config.json'), JSON.generate(ready: true, gpu: 'RTX 5060', vram_gb: 8))
    py = Aire::Env.python
    FileUtils.mkdir_p(File.dirname(py))
    File.write(py, '')
  end

  def test_ui_file_exists_and_initial_state_not_ready
    assert File.exist?(@dialog.file)
    @dialog.trigger('ready')
    state = last_state
    refute state['ready']
    assert_nil state['setup']
    assert_equal [], state['history']
  end

  def test_install_launches_hidden_bootstrap_and_polls
    FileUtils.mkdir_p(Aire::Env.home)
    @dialog.trigger('install')
    exe, args = @spawned.last
    assert_equal 'powershell.exe', exe
    assert_includes args, '-ExecutionPolicy'
    assert_equal 'Hidden', args[args.index('-WindowStyle') + 1]
    assert args.last.end_with?('setup.json')
    assert_equal 'running', last_state['setup']['state']
    refute_empty UI.timers
    @dialog.trigger('install') # pulsar dos veces no lanza dos instalaciones
    assert_equal 1, @spawned.size
  end

  def test_stale_setup_shows_as_interrupted
    file = Aire::Env.setup_progress_file
    FileUtils.mkdir_p(File.dirname(file))
    File.write(file, JSON.generate(state: 'running', updated: Time.now.to_f - 3600, steps: ['x'], step: 0))
    @dialog.trigger('ready')
    assert_equal 'interrupted', last_state['setup']['state']
  end

  def test_dead_installer_process_shows_as_interrupted_quickly
    file = Aire::Env.setup_progress_file
    FileUtils.mkdir_p(File.dirname(file))
    File.write(file, JSON.generate(state: 'running', updated: Time.now.to_f - 30, pid: 4242, steps: ['x'], step: 0))
    original = Aire::Env.method(:process_alive?)
    Aire::Env.define_singleton_method(:process_alive?) { |pid| pid.to_i != 4242 }
    @dialog.trigger('ready')
    assert_equal 'interrupted', last_state['setup']['state']
    # Recién actualizado: aunque no se pueda comprobar el proceso, sigue «running»
    File.write(file, JSON.generate(state: 'running', updated: Time.now.to_f, pid: 4242, steps: ['x'], step: 0))
    @dialog.trigger('ready')
    assert_equal 'running', last_state['setup']['state']
  ensure
    Aire::Env.define_singleton_method(:process_alive?, original) if original
  end

  def test_render_exports_view_and_starts_job
    make_ready
    @dialog.trigger('render', JSON.generate(style: 'nordico', light: 'tarde', quality: 'rapida', variants: 9,
                                            prompt: "con plantas \"verdes\"\ny ñ"))
    exe, args = @spawned.last
    assert exe.end_with?('pythonw.exe')
    assert args[0].end_with?(File.join('backend', 'run.py'))
    assert_equal 'job', args[1]
    opt = ->(k) { args[args.index(k) + 1] }
    assert_equal 'nordico', opt.call('--style')
    assert_equal '4', opt.call('--variants') # limitado a 4
    assert_equal "con plantas \"verdes\"\ny ñ", File.read(opt.call('--prompt-file'), encoding: 'UTF-8')
    scene = JSON.parse(File.read(File.join(opt.call('--export'), 'scene.json')))
    assert_equal 1024, scene['view']['width']
    assert_equal 'running', last_state['job']['state']
  end

  def test_render_empty_view_gives_friendly_message
    make_ready
    Sketchup.active_model = Sketchup::Model.new([], Sketchup.active_model.active_view)
    @dialog.trigger('render', JSON.generate(style: 'modelo', light: 'dia', quality: 'alta', variants: 1, prompt: ''))
    assert(@dialog.scripts.any? { |s| s.include?('No hay nada visible') })
    assert_empty @spawned
  end

  def test_finished_job_appears_in_gallery
    make_ready
    @dialog.trigger('render', JSON.generate(style: 'japandi', light: 'dia', quality: 'alta', variants: 1, prompt: ''))
    job_dir = @win.instance_variable_get(:@job_dir)
    renders = File.join(job_dir, 'renders')
    FileUtils.mkdir_p(renders)
    File.binwrite(File.join(renders, 'render_00_thumb.jpg'), 'JPEGDATA')
    File.write(File.join(renders, 'result.json'), JSON.generate(
      style: 'japandi', light: 'dia', user_prompt: '', created: Time.now.to_f,
      images: [{ path: File.join(renders, 'render_00.png'), thumb: File.join(renders, 'render_00_thumb.jpg') }]
    ))
    File.write(File.join(job_dir, 'job.json'), JSON.generate(state: 'done', updated: Time.now.to_f, steps: [], step: 4))
    UI.timers.last.call
    state = last_state
    assert_equal 'done', state['job']['state']
    assert_equal 1, state['history'].size
    assert state['history'][0]['thumb'].start_with?('data:image/jpeg;base64,')
  end
end
